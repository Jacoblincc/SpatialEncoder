"""
engine_finetune_dynamic.py  —  v0521 sin/cos DoA regression

Multi-task loss
---------------
  total = 1 * CE(classifier)
        + 10 * ( MSE(frame_az_sincos) + MSE(frame_el_sincos) )   ← sin/cos regression
        + 2  * CE(frame_dist)
        + CE(movement_type) + CE(movement_dir) + CE(velocity)

DoA loss变化说明
  旧: CE on 360/180 classes → 不连续，收敛差
  新: MSE on (sin, cos) → 连续平滑，等价于最小化单位圆上的欧氏距离
      weight=10 对应旧方案的 5*(az+el)，保持总梯度量级接近

Evaluation metrics
------------------
  mAP                – mean Average Precision (sound-event classes)
  frame_mae_deg      – mean great-circle angular error (degrees)
  frame_er20         – fraction of frames with DoA error > 20°
  distance_accuracy  – fraction with |pred_bin - gt_bin| ≤ 1  (±0.5 m)
  movement_type/dir/velocity accuracy
"""

import math
import sys
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

import utils.misc as misc
import utils.lr_sched as lr_sched
from utils.stat import calculate_stats, concat_all_gather


# ---------------------------------------------------------------------------
# DoA helpers
# ---------------------------------------------------------------------------

def _az_class_to_sincos(az_class, device):
    """Convert integer azimuth class (0-359 degrees) to (sin, cos) target.

    az_class: (B, N) long tensor, value = degree [0, 359]
    returns : (B, N, 2) float32
    """
    rad = az_class.float() * (math.pi / 180.0)
    return torch.stack([torch.sin(rad), torch.cos(rad)], dim=-1)   # (B, N, 2)


def _el_class_to_sincos(el_class, device):
    """Convert integer elevation class to (sin, cos) target.

    el_class: (B, N) long  — stored as el_deg + 90, so el_deg = el_class - 90
    returns : (B, N, 2) float32
    """
    el_deg = el_class.float() - 90.0
    rad    = el_deg * (math.pi / 180.0)
    return torch.stack([torch.sin(rad), torch.cos(rad)], dim=-1)   # (B, N, 2)


def _sincos_to_deg(sincos):
    """Convert (sin, cos) predictions back to degrees via atan2.

    sincos: numpy array (..., 2)
    returns: degrees in [-180, 180)
    """
    return np.degrees(np.arctan2(sincos[..., 0], sincos[..., 1]))


def _angular_dist_deg_from_sincos(az_sincos_pred, el_sincos_pred, az_gt, el_gt):
    """Great-circle distance from sin/cos predictions and integer gt classes.

    az_sincos_pred: (M, 2)  numpy
    el_sincos_pred: (M, 2)  numpy
    az_gt         : (M,)    numpy  int class [0, 359]
    el_gt         : (M,)    numpy  int class [0, 179]  (el_deg = el_class - 90)

    Returns: (M,) numpy array of angular errors in degrees.
    """
    # Predicted degrees
    az_pred_deg = _sincos_to_deg(az_sincos_pred)            # [-180, 180)
    el_pred_deg = _sincos_to_deg(el_sincos_pred)            # [-90, 90)

    # GT degrees
    az_gt_deg = az_gt.astype(float)
    az_gt_deg[az_gt_deg > 180] -= 360                       # wrap to [-180, 180)
    el_gt_deg = el_gt.astype(float) - 90.0

    az_p = np.deg2rad(az_pred_deg);  az_g = np.deg2rad(az_gt_deg)
    el_p = np.deg2rad(el_pred_deg);  el_g = np.deg2rad(el_gt_deg)

    dot = (np.sin(el_p) * np.sin(el_g) +
           np.cos(el_p) * np.cos(el_g) * np.cos(np.abs(az_p - az_g)))
    dot = np.clip(dot, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    log_writer=None,
    args=None,
):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header     = f'Epoch: [{epoch}]'
    print_freq = 200
    accum_iter = args.accum_iter

    optimizer.zero_grad()
    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, step / len(data_loader) + epoch, args)

        waveforms, label_indices, frame_targets, traj_targets = batch

        waveforms     = waveforms.to(device, non_blocking=True)      # (B, 2, T)
        label_indices = label_indices.to(device, non_blocking=True)  # (B,)  long

        az_gt   = frame_targets['azimuth'].to(device, non_blocking=True)    # (B, N) long
        el_gt   = frame_targets['elevation'].to(device, non_blocking=True)  # (B, N) long
        dist_gt = frame_targets['distance'].to(device, non_blocking=True)   # (B, N) long

        mov_type = traj_targets['movement_type'].to(device, non_blocking=True)
        mov_dir  = traj_targets['movement_direction'].to(device, non_blocking=True)
        velocity = traj_targets['velocity_class'].to(device, non_blocking=True)

        outputs = model(
            waveforms,
            mask_t_prob=args.mask_t_prob,
            mask_f_prob=args.mask_f_prob,
        )

        B, N = az_gt.shape

        # ---- DoA: sin/cos MSE loss ----------------------------------------
        # Convert integer labels → (sin, cos) float targets
        az_sincos_gt = _az_class_to_sincos(az_gt, device)   # (B, N, 2)
        el_sincos_gt = _el_class_to_sincos(el_gt, device)   # (B, N, 2)

        # Normalise prediction onto unit circle before MSE
        # (prevents the network from cheating by shrinking outputs to near-zero)
        az_pred_norm = F.normalize(outputs['frame_az_sincos'], dim=-1)  # (B, N, 2)
        el_pred_norm = F.normalize(outputs['frame_el_sincos'], dim=-1)  # (B, N, 2)

        loss_az = F.mse_loss(az_pred_norm, az_sincos_gt)
        # loss_el = F.mse_loss(el_pred_norm, el_sincos_gt)
        loss_el = F.mse_loss(el_pred_norm, el_sincos_gt) * 5 # 仰角分布集中在水平，MSE天然偏低

        # ---- Distance CE loss --------------------------------------------
        loss_dist = F.cross_entropy(
            outputs['frame_dist'].reshape(B * N, -1),
            dist_gt.reshape(B * N)
        )

        # ---- Classification CE loss --------------------------------------
        loss_cls = criterion(outputs['classifier'], label_indices)

        # ---- Trajectory CE losses ----------------------------------------
        loss_mov_type = F.cross_entropy(outputs['movement_type'], mov_type)
        loss_mov_dir  = F.cross_entropy(outputs['movement_dir'],  mov_dir)
        loss_velocity = F.cross_entropy(outputs['velocity'],      velocity)

        # ---- Total loss --------------------------------------------------
        # DoA weight=10: MSE on (sin,cos) is ~0.5-2.0 range vs CE ~1-4,
        # so weight=10 keeps DoA gradients comparable to the old 5*(CE_az+CE_el).
        loss = (
            1  * loss_cls
            + 10 * loss_az
            + 10 * loss_el
            + 2  * loss_dist
            + loss_mov_type + loss_mov_dir + loss_velocity
        )

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(step + 1) % accum_iter == 0)
        if (step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_cls=loss_cls.item())
        metric_logger.update(loss_az=loss_az.item())
        metric_logger.update(loss_el=loss_el.item())
        metric_logger.update(loss_dist=loss_dist.item())
        metric_logger.update(loss_mov_type=loss_mov_type.item())
        metric_logger.update(loss_mov_dir=loss_mov_dir.item())
        metric_logger.update(loss_velocity=loss_velocity.item())

        max_lr = max(g["lr"] for g in optimizer.param_groups)
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (step + 1) % accum_iter == 0:
            epoch_1000x = int((step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss/total',          loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('loss/cls',            loss_cls.item(),   epoch_1000x)
            log_writer.add_scalar('loss/azimuth',        loss_az.item(),    epoch_1000x)
            log_writer.add_scalar('loss/elev',           loss_el.item(),    epoch_1000x)
            log_writer.add_scalar('loss/distance',       loss_dist.item(),  epoch_1000x)
            log_writer.add_scalar('loss/move_type',      loss_mov_type.item(), epoch_1000x)
            log_writer.add_scalar('loss/move_direction', loss_mov_dir.item(),  epoch_1000x)
            log_writer.add_scalar('loss/velocity',       loss_velocity.item(), epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(data_loader, model, device, dist_eval=False):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()

    all_cls_preds   = []
    all_cls_targets = []

    # per-frame: store raw (sin,cos) predictions and integer gt
    all_az_sincos_preds = []
    all_el_sincos_preds = []
    all_az_gts          = []
    all_el_gts          = []
    all_dist_preds      = []
    all_dist_gts        = []

    # trajectory
    all_mov_type_preds = []
    all_mov_type_gts   = []
    all_mov_dir_preds  = []
    all_mov_dir_gts    = []
    all_vel_preds      = []
    all_vel_gts        = []

    for batch in metric_logger.log_every(data_loader, 200, header):
        waveforms, label_indices, frame_targets, traj_targets = batch

        waveforms     = waveforms.to(device, non_blocking=True)
        label_indices = label_indices.to(device, non_blocking=True)

        outputs = model(waveforms)

        # ---- Classification ------------------------------------------
        cls_pred = outputs['classifier'].detach()
        if dist_eval:
            cls_pred      = concat_all_gather(cls_pred)
            label_indices = concat_all_gather(label_indices)
        all_cls_preds.append(cls_pred.cpu().numpy())
        all_cls_targets.append(label_indices.cpu().numpy())

        # ---- Per-frame DoA (sin/cos predictions, normalised) ---------
        az_sc = F.normalize(outputs['frame_az_sincos'], dim=-1).detach().cpu().numpy()  # (B,N,2)
        el_sc = F.normalize(outputs['frame_el_sincos'], dim=-1).detach().cpu().numpy()  # (B,N,2)
        B, N, _ = az_sc.shape

        all_az_sincos_preds.append(az_sc.reshape(B * N, 2))
        all_el_sincos_preds.append(el_sc.reshape(B * N, 2))
        all_az_gts.append(frame_targets['azimuth'].numpy().flatten())
        all_el_gts.append(frame_targets['elevation'].numpy().flatten())

        # ---- Distance ---------------------------------------------------
        dist_pred = torch.argmax(outputs['frame_dist'], dim=-1).detach().cpu().numpy()
        all_dist_preds.append(dist_pred.flatten())
        all_dist_gts.append(frame_targets['distance'].numpy().flatten())

        # ---- Trajectory labels -----------------------------------------
        all_mov_type_preds.append(torch.argmax(outputs['movement_type'], dim=-1).cpu().numpy())
        all_mov_type_gts.append(traj_targets['movement_type'].numpy())
        all_mov_dir_preds.append(torch.argmax(outputs['movement_dir'],   dim=-1).cpu().numpy())
        all_mov_dir_gts.append(traj_targets['movement_direction'].numpy())
        all_vel_preds.append(torch.argmax(outputs['velocity'], dim=-1).cpu().numpy())
        all_vel_gts.append(traj_targets['velocity_class'].numpy())

    # ----------------------------------------------------------------
    # mAP
    # ----------------------------------------------------------------
    cls_preds   = np.concatenate(all_cls_preds,   axis=0)
    cls_targets = np.concatenate(all_cls_targets, axis=0)

    num_classes        = cls_preds.shape[1]
    cls_targets_onehot = np.eye(num_classes)[cls_targets]
    stats  = calculate_stats(cls_preds, cls_targets_onehot)
    ap_vals = [s['AP'] for s in stats if not np.isnan(s['AP'])]
    mAP    = float(np.mean(ap_vals)) if ap_vals else 0.0

    # ----------------------------------------------------------------
    # Per-frame DoA metrics (decoded from sin/cos)
    # ----------------------------------------------------------------
    az_sc_all = np.concatenate(all_az_sincos_preds, axis=0)   # (M, 2)
    el_sc_all = np.concatenate(all_el_sincos_preds, axis=0)   # (M, 2)
    az_gt_all = np.concatenate(all_az_gts)                    # (M,)
    el_gt_all = np.concatenate(all_el_gts)                    # (M,)

    angular_errors = _angular_dist_deg_from_sincos(
        az_sc_all, el_sc_all, az_gt_all, el_gt_all
    )
    frame_mae  = float(np.mean(angular_errors))
    frame_er20 = float(np.mean(angular_errors > 20.0))

    # ----------------------------------------------------------------
    # Distance accuracy
    # ----------------------------------------------------------------
    dist_pred_all = np.concatenate(all_dist_preds)
    dist_gt_all   = np.concatenate(all_dist_gts)
    dist_acc = float(np.sum(np.abs(dist_pred_all - dist_gt_all) <= 1)) / len(dist_gt_all)

    # ----------------------------------------------------------------
    # Trajectory classification accuracy
    # ----------------------------------------------------------------
    mov_type_preds = np.concatenate(all_mov_type_preds)
    mov_type_gts   = np.concatenate(all_mov_type_gts)
    mov_dir_preds  = np.concatenate(all_mov_dir_preds)
    mov_dir_gts    = np.concatenate(all_mov_dir_gts)
    vel_preds      = np.concatenate(all_vel_preds)
    vel_gts        = np.concatenate(all_vel_gts)

    mov_type_acc = float(np.mean(mov_type_preds == mov_type_gts))
    mov_dir_acc  = float(np.mean(mov_dir_preds  == mov_dir_gts))
    velocity_acc = float(np.mean(vel_preds       == vel_gts))

    print(f"mAP: {mAP:.4f}")
    print(f"Frame DoA MAE: {frame_mae:.2f}° | ER20: {frame_er20:.4f}")
    print(f"Distance accuracy (±0.5m): {dist_acc:.4f}")
    print(f"Movement type acc: {mov_type_acc:.4f} | dir acc: {mov_dir_acc:.4f} | velocity acc: {velocity_acc:.4f}")

    return {
        "mAP":               mAP,
        "frame_mae_deg":     frame_mae,
        "frame_er20":        frame_er20,
        "distance_accuracy": dist_acc,
        "movement_type_acc": mov_type_acc,
        "movement_dir_acc":  mov_dir_acc,
        "velocity_acc":      velocity_acc,
    }