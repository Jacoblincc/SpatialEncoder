"""
main_finetune_dynamic.py

Main training / evaluation script for DynamicSpatialAST.

Typical usage (single GPU, eval only):
    python main_finetune_dynamic.py \
        --spatial_ast_ckpt /path/to/finetuned.pth \
        --dynamic_train_json /path/to/train_dynamic.json \
        --dynamic_eval_json  /path/to/eval_dynamic.json \
        --dynamic_audio_root /path/to/dynamic_dataset/audio \
        --label_csv /path/to/class_labels_indices_subset.csv \
        --eval

Distributed training (4 GPUs, Stage 1 – freeze SpatialAST):
    python -m torch.distributed.launch --nproc_per_node=4 --use_env \
        main_finetune_dynamic.py \
        --spatial_ast_ckpt /path/to/finetuned.pth \
        --dynamic_train_json /path/to/train_dynamic.json \
        --dynamic_eval_json  /path/to/eval_dynamic.json \
        --dynamic_audio_root /path/to/dynamic_dataset/audio \
        --label_csv /path/to/class_labels_indices_subset.csv \
        --freeze_spatial_ast \
        --epochs 30 --batch_size 8 --blr 1e-3

Stage 2 (unfreeze last 4 blocks):
    python -m torch.distributed.launch ... \
        --resume /path/to/stage1_checkpoint.pth \
        --unfreeze_last_n_blocks 4 \
        --epochs 20 --blr 5e-5
"""

import datetime
import json
import os
import time
from pathlib import Path

import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import timm
assert timm.__version__ == "0.3.2"

import utils.lr_decay as lrd
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler

import spatial_ast as spatial_ast_module
from dynamic_spatial_ast import DynamicSpatialAST
from data.dynamic_dataset import DynamicSpatialDataset
from engine_finetune_dynamic import evaluate, train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('DynamicSpatialAST fine-tuning', add_help=False)

    # ---- optimisation --------------------------------------------------
    parser.add_argument('--batch_size',   default=8,   type=int)
    parser.add_argument('--epochs',       default=30,  type=int)
    parser.add_argument('--accum_iter',   default=1,   type=int)
    parser.add_argument('--clip_grad',    default=None, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--lr',           default=None, type=float)
    parser.add_argument('--blr',          default=1e-3, type=float,
                        help='base lr: actual_lr = blr * batch_size / 256')
    parser.add_argument('--lr_backbone',  default=1e-5, type=float,
                        help='LR for the SpatialAST backbone (when unfrozen)')
    parser.add_argument('--min_lr',       default=1e-6, type=float)
    parser.add_argument('--warmup_epochs',default=4,   type=int)
    parser.add_argument('--layer_decay',  default=0.75, type=float)

    # ---- model ---------------------------------------------------------
    parser.add_argument('--spatial_ast_ckpt', default='', type=str,
                        help='Path to pretrained SpatialAST checkpoint (finetuned.pth)')
    parser.add_argument('--nb_classes',       default=355, type=int)
    parser.add_argument('--n_segments',       default=10,  type=int,
                        help='Number of temporal segments per sample')
    parser.add_argument('--seg_duration',     default=1.0, type=float)
    parser.add_argument('--n_temporal_layers',default=2,   type=int)
    parser.add_argument('--n_temporal_heads', default=8,   type=int)
    parser.add_argument('--freeze_spatial_ast', action='store_true', default=False,
                        help='Stage 1: freeze SpatialAST; train only temporal + heads')
    parser.add_argument('--unfreeze_last_n_blocks', default=0, type=int,
                        help='Stage 2: unfreeze the last N ViT blocks of SpatialAST')

    # ---- masking (passed to model.forward) ----------------------------
    parser.add_argument('--mask_t_prob', default=0.0, type=float)
    parser.add_argument('--mask_f_prob', default=0.0, type=float)

    # ---- data ----------------------------------------------------------
    parser.add_argument('--dynamic_train_json', default='', type=str)
    parser.add_argument('--dynamic_eval_json',  default='', type=str)
    parser.add_argument('--dynamic_audio_root', default='', type=str,
                        help='Root directory for audio files referenced in JSON')
    parser.add_argument('--label_csv', default='', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True)
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    # ---- misc ----------------------------------------------------------
    parser.add_argument('--output_dir',  default='./outputs_dynamic')
    parser.add_argument('--log_dir',     default='./outputs_dynamic')
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--seed',        default=0, type=int)
    parser.add_argument('--resume',      default='', type=str)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--eval',        action='store_true')
    parser.add_argument('--dist_eval',   action='store_true', default=False)
    parser.add_argument('--first_eval_ep', default=0, type=int)

    # ---- distributed ---------------------------------------------------
    parser.add_argument('--world_size',  default=1,   type=int)
    parser.add_argument('--local_rank',  default=-1,  type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url',    default='env://')

    return parser


def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed   = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # ---- datasets ------------------------------------------------------
    dataset_train = DynamicSpatialDataset(
        json_path    = args.dynamic_train_json,
        audio_root   = args.dynamic_audio_root,
        label_csv    = args.label_csv,
        n_segments   = args.n_segments,
        seg_duration = args.seg_duration,
        normalize    = True,
        mode         = 'train',
    )
    dataset_val = DynamicSpatialDataset(
        json_path    = args.dynamic_eval_json,
        audio_root   = args.dynamic_audio_root,
        label_csv    = args.label_csv,
        n_segments   = args.n_segments,
        seg_duration = args.seg_duration,
        normalize    = True,
        mode         = 'eval',
    )

    num_tasks    = misc.get_world_size()
    global_rank  = misc.get_rank()

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    if args.dist_eval:
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=True,
        collate_fn=DynamicSpatialDataset.collate_fn,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=args.pin_mem, drop_last=False,
        collate_fn=DynamicSpatialDataset.collate_fn,
    )

    # ---- build base SpatialAST ----------------------------------------
    from functools import partial
    # LXWJE 初始化spatial ast基础模型
    base_model = spatial_ast_module.build_AST(
        num_classes=args.nb_classes,
        drop_path_rate=0.1,
        num_cls_tokens=3,
    )
    if args.spatial_ast_ckpt:
        print(f"Loading SpatialAST checkpoint: {args.spatial_ast_ckpt}")
        ckpt = torch.load(args.spatial_ast_ckpt, map_location='cpu')
        state = ckpt.get('model', ckpt)
        msg = base_model.load_state_dict(state, strict=False)
        print(msg)

    # ---- build DynamicSpatialAST --------------------------------------
    # LXWJE 加载模型，在spatial ast基础上加了一个时序transformer。详细看 dynamic_spatial_ast.py
    model = DynamicSpatialAST(
        spatial_ast          = base_model,
        n_segments           = args.n_segments,
        d_model              = 768,
        n_temporal_heads     = args.n_temporal_heads,
        n_temporal_layers    = args.n_temporal_layers,
        num_cls              = args.nb_classes,
        freeze_spatial_ast   = args.freeze_spatial_ast,
        unfreeze_last_n_blocks = args.unfreeze_last_n_blocks,
    )

    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"Trainable: {n}  {list(p.shape)}")

    model.to(device)
    model_without_ddp = model
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params / 1e6:.2f}M")

    eff_batch = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch / 256
    print(f"Effective batch size: {eff_batch}, lr: {args.lr:.2e}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # ---- optimiser (differential LR for backbone vs. new heads) -------
    if args.unfreeze_last_n_blocks > 0:
        param_groups = model_without_ddp.get_param_groups(
            lr_backbone=args.lr_backbone, lr_head=args.lr
        )
    else:
        # stage 1: only new parameters, single LR
        param_groups = [{'params': [p for p in model.parameters() if p.requires_grad],
                         'lr': args.lr}]

    optimizer    = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    loss_scaler  = NativeScaler()
    # criterion    = nn.BCEWithLogitsLoss() # TODO LXWJE 这是多分类的loss，暂时修改成单分类！！！
    criterion    = nn.CrossEntropyLoss() # LXWJE 这是多分类的loss，暂时修改成单分类！！！
    

    # LXWJE 从这里面resume from ckpt
    misc.load_model(args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler
                    )

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args.dist_eval)
        for k, v in test_stats.items():
            print(f"  {k}: {v:.4f}")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_mae   = float('inf')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=args.clip_grad,
            log_writer=log_writer,
            args=args,
        )

        if args.output_dir and (epoch % 5 == 0 or epoch == args.epochs - 1):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
            )

        if epoch >= args.first_eval_ep:
            test_stats = evaluate(data_loader_val, model, device, args.dist_eval)
            best_mae   = min(best_mae, test_stats.get('frame_mae_deg', float('inf')))
            print(f"Best frame MAE so far: {best_mae:.2f}°")
        else:
            test_stats = {}
            print("Too early to evaluate.")

        if log_writer is not None:
            for k, v in test_stats.items():
                log_writer.add_scalar(f'perf/{k}', v, epoch)

        log_stats = {
            'epoch': epoch,
            'n_parameters': n_params,
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}':  v for k, v in test_stats.items()},
        }
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {total_time_str}")


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
