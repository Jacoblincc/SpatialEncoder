#!/usr/bin/env python3
"""
生成动态空间音频数据集（听感强化版）
- 运动幅度、速度显著提升
- 距离响度调制
- 混响降低，直达声突出
- 标签体系：5粗粒度运动类型 + 15方向 + 3速度档

python /home/master/xlin/Spatial/Spatial-AST-main/data/generate_dynamic_data_plus_mp_v0515.py \
    --meta_json     /home/master/xlin/datasets/SpatialAudio/SpatialSoundQA/AudioSet/metadata/my_process/json/sample50_10.json \
    --audio_prefix  /home/master/xlin/datasets/SpatialAudio/SpatialSoundQA/AudioSet \
    --output_dir    /home/master/xlin/Spatial/Spatial-AST-main/data/v0515-50-10-full \
    --split_name    train \
    --n_per_item    20 \
    --num_workers   8

"""

import argparse, json, math, os, random
from pathlib import Path
import numpy as np
import soundfile as sf
import torchaudio, torch
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EAR_DIST   = 0.17
SR         = 32_000
DURATION_S = 10.0
TARGET_LEN = int(SR * DURATION_S)

# ---------------------------------------------------------------------------
# 3D direction label (15 classes)
# ---------------------------------------------------------------------------
_CUBE_DIRS = [
    (0, 1, 0), (0,-1, 0), (1, 0, 0), (-1,0, 0),
    (0, 0, 1), (0, 0,-1),
    (-1, 1, 1), ( 1, 1, 1), (-1,-1, 1), ( 1,-1, 1),
    (-1, 1,-1), ( 1, 1,-1), (-1,-1,-1), ( 1,-1,-1),
]
_CUBE_DIR_NAMES = [
    "front", "back", "right", "left", "up", "down",
    "front-left-up", "front-right-up", "back-left-up", "back-right-up",
    "front-left-down", "front-right-down", "back-left-down", "back-right-down",
]

def direction_label_3d(displacement, threshold=0.05):
    dx, dy, dz = float(displacement[0]), float(displacement[1]), float(displacement[2])
    mag = math.sqrt(dx**2 + dy**2 + dz**2)
    if mag < threshold:
        return "none"
    vec = np.array([dx, dy, dz]) / mag
    best_idx = 0
    best_dot = -1.0
    for idx, ref in enumerate(_CUBE_DIRS):
        dot = vec[0]*ref[0] + vec[1]*ref[1] + vec[2]*ref[2]
        if dot > best_dot:
            best_dot = dot
            best_idx = idx
    return _CUBE_DIR_NAMES[best_idx]

# ---------------------------------------------------------------------------
# Movement types (10 fine, 5 coarse)
# ---------------------------------------------------------------------------
MOVEMENT_TYPES = [
    "static", "linear_arbitary",
    "arc_horizontal", "arc_vertical",
    "helix_up", "helix_down",
    "oscillation_lateral", "oscillation_depth", "oscillation_vertical",
    "random_walk_smooth",
]
COARSE_MAP = {
    "static": "static", "linear_arbitary": "linear",
    "arc_horizontal": "arc", "arc_vertical": "arc",
    "helix_up": "arc", "helix_down": "arc",
    "oscillation_lateral": "oscillation", "oscillation_depth": "oscillation",
    "oscillation_vertical": "oscillation", "random_walk_smooth": "random",
}

def coarse_movement_type(fine): return COARSE_MAP[fine]

# ---------------------------------------------------------------------------
# Room helpers (强化：减小RT60，降低max_order)
# ---------------------------------------------------------------------------
def _detect_has_voice(wav_path: str) -> bool:
    try:
        import webrtcvad, struct
        wav, sr = sf.read(wav_path, always_2d=False)
        if wav.ndim > 1: wav = wav[:, 0]
        target_sr, frame_ms = 16000, 30
        frame_len = int(target_sr * frame_ms / 1000)
        if sr != target_sr:
            wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
            wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
            wav = wav_t.squeeze(0).numpy()
        wav_int16 = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
        vad = webrtcvad.Vad(2)
        voice_frames = sum(1 for start in range(0, len(wav_int16)-frame_len, frame_len)
                           if vad.is_speech(struct.pack(f"{frame_len}h", *wav_int16[start:start+frame_len]), target_sr))
        total_frames = max(1, (len(wav_int16) - frame_len) // frame_len)
        return (voice_frames / total_frames) > 0.3
    except: return False

def _sample_room_params(has_voice: bool) -> dict:
    if has_voice:
        room_dims = [random.uniform(3.0, 6.0), random.uniform(3.0, 6.0), random.uniform(2.4, 3.0)]
        rt60 = random.uniform(0.1, 0.3)
        max_order = random.randint(2, 5)
    else:
        room_dims = [random.uniform(4.0, 10.0), random.uniform(4.0, 10.0), random.uniform(2.5, 3.5)]
        rt60 = random.uniform(0.1, 0.4)          # 原 0.2~0.8 → 0.1~0.4
        max_order = random.randint(3, 6)         # 原 10 → 5 左右
    receiver_pos = [room_dims[0]/2 + random.uniform(-0.5, 0.5),
                    room_dims[1]/2 + random.uniform(-0.5, 0.5),
                    1.2]
    return {'room_dims': room_dims, 'receiver_pos': receiver_pos, 'rt60': rt60, 'max_order': max_order}

def _rt60_to_absorption(rt60, room_dims):
    V = room_dims[0]*room_dims[1]*room_dims[2]
    S = 2*(room_dims[0]*room_dims[1] + room_dims[1]*room_dims[2] + room_dims[0]*room_dims[2])
    return float(np.clip(0.161*V/(rt60*S), 0.01, 0.99))

def compute_binaural_rir(room_dims, source_pos, receiver_pos, rt60=0.4, sr=SR, max_order=10):
    import pyroomacoustics as pra
    room_dims = np.array(room_dims, dtype=float)
    source_pos = np.array(source_pos, dtype=float)
    receiver_pos = np.array(receiver_pos, dtype=float)
    margin = 0.2
    source_pos = np.clip(source_pos, margin, room_dims-margin)
    receiver_pos = np.clip(receiver_pos, margin+EAR_DIST/2, room_dims-margin-EAR_DIST/2)
    absorption = _rt60_to_absorption(rt60, room_dims.tolist())
    room = pra.ShoeBox(room_dims, fs=sr, materials=pra.Material(absorption),
                       max_order=max_order, ray_tracing=False, air_absorption=True)
    left_ear  = receiver_pos + np.array([-EAR_DIST/2, 0, 0])
    right_ear = receiver_pos + np.array([+EAR_DIST/2, 0, 0])
    room.add_microphone(np.stack([left_ear, right_ear], axis=1))
    room.add_source(source_pos.tolist())
    room.image_source_model()
    room.compute_rir()
    rir_L = np.array(room.rir[0][0], dtype=np.float32)
    rir_R = np.array(room.rir[1][0], dtype=np.float32)
    max_len = max(len(rir_L), len(rir_R))
    rir_L = np.pad(rir_L, (0, max_len-len(rir_L)))
    rir_R = np.pad(rir_R, (0, max_len-len(rir_R)))
    return np.stack([rir_L, rir_R], axis=0)

# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------
def xyz_to_spatial_labels(source_pos, receiver_pos):
    diff = np.array(source_pos) - np.array(receiver_pos)
    dist_m = float(np.linalg.norm(diff))
    az_deg = math.degrees(math.atan2(diff[0], diff[1])) % 360
    az_class = int(round(az_deg)) % 360
    horiz = math.sqrt(diff[0]**2 + diff[1]**2)
    el_deg = math.degrees(math.atan2(diff[2], horiz))
    el_class = int(round(el_deg + 90)) % 180
    dist_class = min(int(round(dist_m * 2)), 20)
    return az_class, el_class, dist_class, dist_m

def _velocity_class(speed_mps: float) -> int:
    if speed_mps < 0.5: return 0
    elif speed_mps < 1.5: return 1
    else: return 2

# ---------------------------------------------------------------------------
# Trajectory generators (参数强化)
# ---------------------------------------------------------------------------
def _rand_start_pos(room_dims, receiver_pos, min_dist=1.5, max_dist=4.0, z_range=None):
    rd = np.array(room_dims)
    rp = np.array(receiver_pos)
    z_lo = z_range[0] if z_range else 0.15
    z_hi = z_range[1] if z_range else rd[2] - 0.15
    for _ in range(300):
        pos = np.array([random.uniform(0.15, rd[0]-0.15),
                        random.uniform(0.15, rd[1]-0.15),
                        random.uniform(z_lo, z_hi)])
        if min_dist <= np.linalg.norm(pos - rp) <= max_dist:
            return pos.tolist()
    angle = random.uniform(0, 2*math.pi)
    dist = random.uniform(min_dist, min(max_dist, min(rd[:2])/2 - 0.2))
    x = np.clip(rp[0] + dist*math.sin(angle), 0.15, rd[0]-0.15)
    y = np.clip(rp[1] + dist*math.cos(angle), 0.15, rd[1]-0.15)
    z = np.clip(rp[2], z_lo, z_hi)
    return [float(x), float(y), float(z)]

def _traj_static(n_steps, room_dims, receiver_pos):
    start = _rand_start_pos(room_dims, receiver_pos)
    return [start]*n_steps, "static", "none", 0

def _traj_linear_arbitary(n_steps, room_dims, receiver_pos):
    rd, rp = np.array(room_dims), np.array(receiver_pos)
    margin = 0.15
    start = _rand_start_pos(room_dims, receiver_pos, min_dist=1.5, max_dist=4.0)
    theta = random.uniform(0, 2*math.pi)
    phi = random.uniform(-math.pi/3, math.pi/3)
    direction = np.array([math.cos(phi)*math.cos(theta), math.cos(phi)*math.sin(theta), math.sin(phi)])
    total_dist = random.uniform(3.0, 8.0)          # 原 1.5~6 → 3~8
    step_dist = total_dist / n_steps
    positions = []
    for i in range(n_steps):
        pos = np.clip(np.array(start) + direction*step_dist*i, margin, rd-margin)
        positions.append(pos.tolist())
    net = np.array(positions[-1]) - np.array(positions[0])
    dir_label = direction_label_3d(net)
    vc = _velocity_class(step_dist)
    return positions, "linear_arbitary", dir_label, vc

def _traj_arc_horizontal(n_steps, room_dims, receiver_pos):
    rd, rp = np.array(room_dims), np.array(receiver_pos)
    margin = 0.15
    dist_r = random.uniform(2.0, 6.0)             # 原 1.8~5 → 2~6
    start_angle = random.uniform(0, 2*math.pi)
    arc_span = random.uniform(math.pi/2, math.pi*1.8)  # 角度跨度更大
    dir_sign = random.choice([1, -1])
    positions = []
    for i in range(n_steps):
        t = i / max(n_steps-1, 1)
        angle = start_angle + dir_sign * arc_span * t
        pos = np.clip([rp[0] + dist_r*math.sin(angle),
                       rp[1] + dist_r*math.cos(angle),
                       rp[2]], margin, rd-margin)
        positions.append(pos.tolist())
    net = np.array(positions[-1]) - np.array(positions[0])
    dir_label = direction_label_3d(net)
    arc_len = dist_r * arc_span
    vc = _velocity_class(arc_len / n_steps)
    return positions, "arc_horizontal", dir_label, vc

def _traj_arc_vertical(n_steps, room_dims, receiver_pos):
    rd, rp = np.array(room_dims), np.array(receiver_pos)
    margin = 0.15
    plane = random.choice(['xz', 'yz'])
    dist_r = random.uniform(2.0, 5.0)
    start_angle = random.uniform(0, 2*math.pi)
    arc_span = random.uniform(math.pi/2, math.pi*1.5)
    dir_sign = random.choice([1, -1])
    center = rp.copy()
    positions = []
    for i in range(n_steps):
        t = i / max(n_steps-1, 1)
        angle = start_angle + dir_sign * arc_span * t
        if plane == 'xz':
            x = center[0] + dist_r*math.cos(angle)
            z = center[2] + dist_r*math.sin(angle)
            y = center[1]
        else:
            y = center[1] + dist_r*math.cos(angle)
            z = center[2] + dist_r*math.sin(angle)
            x = center[0]
        pos = np.clip([x, y, z], margin, rd-margin)
        positions.append(pos.tolist())
    net = np.array(positions[-1]) - np.array(positions[0])
    dir_label = direction_label_3d(net)
    arc_len = dist_r * arc_span
    vc = _velocity_class(arc_len / n_steps)
    return positions, "arc_vertical", dir_label, vc

def _gen_helix(n_steps, room_dims, receiver_pos, upward):
    rd, rp = np.array(room_dims), np.array(receiver_pos)
    margin = 0.15
    radius = random.uniform(2.0, 5.0)            # 原 1.5~4
    turns = random.uniform(0.5, 2.0)
    total_z = random.uniform(2.0, 4.0) * (1 if upward else -1)  # 原 1~2.5
    start_angle = random.uniform(0, 2*math.pi)
    dir_sign = random.choice([1, -1])
    positions = []
    for i in range(n_steps):
        t = i / max(n_steps-1, 1)
        angle = start_angle + dir_sign * 2*math.pi * turns * t
        x = rp[0] + radius*math.sin(angle)
        y = rp[1] + radius*math.cos(angle)
        z = rp[2] + total_z * t
        pos = np.clip([x, y, z], margin, rd-margin)
        positions.append(pos.tolist())
    net = np.array(positions[-1]) - np.array(positions[0])
    dir_label = direction_label_3d(net)
    h_arc = radius * 2*math.pi * turns
    v_dist = abs(total_z)
    path_len = math.sqrt(h_arc**2 + v_dist**2)
    vc = _velocity_class(path_len / n_steps)
    traj_type = "helix_up" if upward else "helix_down"
    return positions, traj_type, dir_label, vc

def _traj_helix_up(n_steps, room_dims, receiver_pos):
    return _gen_helix(n_steps, room_dims, receiver_pos, upward=True)
def _traj_helix_down(n_steps, room_dims, receiver_pos):
    return _gen_helix(n_steps, room_dims, receiver_pos, upward=False)

def _gen_oscillation(n_steps, axis, room_dims, receiver_pos):
    rd, rp = np.array(room_dims), np.array(receiver_pos)
    margin = 0.15
    amplitude = random.uniform(1.0, 3.0)          # 原 0.6~1.8 → 1~3
    step = random.uniform(0.3, 0.8)               # 原 0.08~0.25 → 0.3~0.8 m/s
    center = np.array(_rand_start_pos(room_dims, receiver_pos, min_dist=1.5, max_dist=4.0))
    low_b = center[axis] - amplitude
    high_b = center[axis] + amplitude
    if low_b < margin: center[axis] += margin - low_b
    if high_b > rd[axis] - margin: center[axis] -= high_b - (rd[axis] - margin)
    period = 4 * amplitude
    positions = []
    d = 0.0
    for i in range(n_steps):
        mod = d % period
        if mod <= 2*amplitude:
            val = -amplitude + mod
        else:
            val = 3*amplitude - mod
        pos = center.copy()
        pos[axis] = center[axis] + val
        pos = np.clip(pos, margin, rd-margin)
        positions.append(pos.tolist())
        d += step
    net = np.array(positions[-1]) - np.array(positions[0])
    dir_label = direction_label_3d(net)
    vc = _velocity_class(step)
    return positions, dir_label, vc, step

def _traj_oscillation_lateral(n_steps, room_dims, receiver_pos):
    pos, dl, vc, _ = _gen_oscillation(n_steps, 0, room_dims, receiver_pos)
    return pos, "oscillation_lateral", dl, vc
def _traj_oscillation_depth(n_steps, room_dims, receiver_pos):
    pos, dl, vc, _ = _gen_oscillation(n_steps, 1, room_dims, receiver_pos)
    return pos, "oscillation_depth", dl, vc
def _traj_oscillation_vertical(n_steps, room_dims, receiver_pos):
    pos, dl, vc, _ = _gen_oscillation(n_steps, 2, room_dims, receiver_pos)
    return pos, "oscillation_vertical", dl, vc

def _traj_random_walk_smooth(n_steps, room_dims, receiver_pos):
    rd, rp = np.array(room_dims), np.array(receiver_pos)
    margin = 0.15
    step_size = random.uniform(0.4, 1.0)          # 原 0.15~0.5 → 0.4~1.0
    start = _rand_start_pos(room_dims, receiver_pos, min_dist=1.5, max_dist=4.0)
    positions = [start]
    current = np.array(start, dtype=float)
    theta = random.uniform(0, 2*math.pi)
    phi = random.uniform(-math.pi/3, math.pi/3)
    dir_vec = np.array([math.cos(phi)*math.cos(theta), math.cos(phi)*math.sin(theta), math.sin(phi)])
    for _ in range(n_steps-1):
        theta += random.gauss(0, 0.3)
        phi = np.clip(phi + random.gauss(0, 0.15), -math.pi/2, math.pi/2)
        dir_vec = np.array([math.cos(phi)*math.cos(theta), math.cos(phi)*math.sin(theta), math.sin(phi)])
        dir_vec /= np.linalg.norm(dir_vec)
        new_pos = current + dir_vec * step_size
        new_pos = np.clip(new_pos, margin, rd-margin)
        positions.append(new_pos.tolist())
        current = new_pos
    net = np.array(positions[-1]) - np.array(positions[0])
    dl = direction_label_3d(net)
    vc = _velocity_class(step_size)
    return positions, "random_walk_smooth", dl, vc

_TRAJ_FN = {
    "static": _traj_static, "linear_arbitary": _traj_linear_arbitary,
    "arc_horizontal": _traj_arc_horizontal, "arc_vertical": _traj_arc_vertical,
    "helix_up": _traj_helix_up, "helix_down": _traj_helix_down,
    "oscillation_lateral": _traj_oscillation_lateral,
    "oscillation_depth": _traj_oscillation_depth,
    "oscillation_vertical": _traj_oscillation_vertical,
    "random_walk_smooth": _traj_random_walk_smooth,
}

def generate_trajectory(traj_type, n_steps, room_dims, receiver_pos):
    positions, mov_type, mov_dir, vc = _TRAJ_FN[traj_type](n_steps, room_dims, receiver_pos)
    return positions, mov_type, mov_dir, vc, coarse_movement_type(mov_type)

# ---------------------------------------------------------------------------
# Frame labels & trajectory summary
# ---------------------------------------------------------------------------
def compute_frame_labels(positions, receiver_pos):
    labels = []
    for i, pos in enumerate(positions):
        az, el, dist_cls, dist_m = xyz_to_spatial_labels(pos, receiver_pos)
        if i > 0:
            prev = positions[i-1]
            az_prev, *_ = xyz_to_spatial_labels(prev, receiver_pos)
            az_delta = float((az - az_prev + 180) % 360 - 180)
            el_prev = xyz_to_spatial_labels(prev, receiver_pos)[1]
            el_delta = float(el - el_prev)
            dist_prev = xyz_to_spatial_labels(prev, receiver_pos)[3]
            dist_delta = float(dist_m - dist_prev)
            speed_mps = float(np.linalg.norm(np.array(pos) - np.array(prev)))
        else:
            az_delta = el_delta = dist_delta = speed_mps = 0.0
        radial = "stable" if abs(dist_delta) < 0.1 else ("receding" if dist_delta > 0 else "approaching")
        angular = "stable" if abs(az_delta) < 2.0 else ("clockwise" if az_delta > 0 else "counter_clockwise")
        if i > 0:
            dz = float(np.array(pos)[2] - np.array(positions[i-1])[2])
            vertical = "stable" if abs(dz) < 0.05 else ("ascending" if dz > 0 else "descending")
        else:
            vertical = "stable"
        labels.append({
            "t": i, "position": [round(float(v), 3) for v in pos],
            "azimuth": az, "elevation": el, "distance": dist_cls,
            "distance_m": round(dist_m, 3),
            "az_delta": round(az_delta,2), "el_delta": round(el_delta,2),
            "dist_delta": round(dist_delta,3), "speed_mps": round(speed_mps,3),
            "velocity_class": _velocity_class(speed_mps),
            "radial_motion": radial, "angular_motion": angular,
            "vertical_motion": vertical,
        })
    return labels

def compute_trajectory_summary(positions, receiver_pos, movement_type,
                               movement_direction, velocity_class, movement_category):
    positions = np.array(positions)
    receiver = np.array(receiver_pos)
    distances = [float(np.linalg.norm(p - receiver)) for p in positions]
    azimuths = [xyz_to_spatial_labels(p.tolist(), receiver_pos)[0] for p in positions]
    elevations = [xyz_to_spatial_labels(p.tolist(), receiver_pos)[1] for p in positions]
    total_path = float(sum(np.linalg.norm(positions[i+1]-positions[i]) for i in range(len(positions)-1)))
    az_arr = np.array(azimuths, dtype=float)
    az_span = float(np.max(az_arr) - np.min(az_arr))
    if az_span > 180: az_span = 360 - az_span
    return {
        "movement_type": movement_type, "movement_category": movement_category,
        "movement_direction": movement_direction, "velocity_class": velocity_class,
        "total_path_m": round(total_path,3), "avg_speed_mps": round(total_path/max(len(positions)-1,1),3),
        "start_pos": [round(float(v),3) for v in positions[0]],
        "end_pos": [round(float(v),3) for v in positions[-1]],
        "min_distance_m": round(min(distances),3), "max_distance_m": round(max(distances),3),
        "mean_distance_m": round(float(np.mean(distances)),3),
        "az_span_deg": round(az_span,2), "el_span_deg": round(float(max(elevations)-min(elevations)),2),
        "has_z_motion": bool(float(np.std(positions[:,2])) > 0.05),
    }

# ---------------------------------------------------------------------------
# RIR crossfade + distance‑based gain
# ---------------------------------------------------------------------------
def _apply_rir_crossfade(dry, rirs, seg_len, sr, distances=None):
    """新增 distances 参数，用于逐段增益调制"""
    from scipy.signal import fftconvolve
    n_seg = len(rirs)
    max_rir_len = max(r.shape[1] for r in rirs)
    rirs_padded = [np.pad(r, ((0,0),(0,max_rir_len-r.shape[1]))) for r in rirs]
    total = n_seg * seg_len
    out_len = total + max_rir_len + seg_len
    output_L = np.zeros(out_len, dtype=np.float32)
    output_R = np.zeros(out_len, dtype=np.float32)
    fade_samples = int(seg_len * 0.2)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)

    # 如果提供了距离，计算每段增益（距离反比，钳位避免过响）
    if distances is not None:
        gains = 1.0 / (np.array(distances) + 0.3)   # 加常数防止近处爆炸
        gains = np.clip(gains, 0.5, 3.0)            # 限制增益范围
    else:
        gains = np.ones(n_seg, dtype=np.float32)

    for i in range(n_seg):
        chunk = dry[i*seg_len:(i+1)*seg_len] * gains[i]  # 应用距离增益
        rir_cur = rirs_padded[i]
        rir_next = rirs_padded[(i+1) % n_seg]
        conv_L_cur = fftconvolve(chunk, rir_cur[0]).astype(np.float32)
        conv_R_cur = fftconvolve(chunk, rir_cur[1]).astype(np.float32)
        conv_L_nxt = fftconvolve(chunk, rir_next[0]).astype(np.float32)
        conv_R_nxt = fftconvolve(chunk, rir_next[1]).astype(np.float32)
        min_len = min(len(conv_L_cur), len(conv_L_nxt))
        conv_L_cur = conv_L_cur[:min_len]; conv_R_cur = conv_R_cur[:min_len]
        conv_L_nxt = conv_L_nxt[:min_len]; conv_R_nxt = conv_R_nxt[:min_len]
        conv_L = conv_L_cur.copy(); conv_R = conv_R_cur.copy()
        fs, fe = seg_len - fade_samples, seg_len - fade_samples + min(fade_samples, min_len - (seg_len - fade_samples))
        if fe > fs:
            conv_L[fs:fe] = conv_L_cur[fs:fe]*fade_out[:fe-fs] + conv_L_nxt[fs:fe]*fade_in[:fe-fs]
            conv_R[fs:fe] = conv_R_cur[fs:fe]*fade_out[:fe-fs] + conv_R_nxt[fs:fe]*fade_in[:fe-fs]
        start = i*seg_len
        end = min(start+len(conv_L), out_len)
        output_L[start:end] += conv_L[:end-start]
        output_R[start:end] += conv_R[:end-start]
    return output_L[:total], output_R[:total]

# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------
def _load_and_prepare_audio(wav_path, target_sr=SR, target_len=TARGET_LEN):
    wav, sr = sf.read(wav_path, always_2d=False)
    if wav.ndim > 1: wav = wav[:, 0]
    if sr != target_sr:
        wav_t = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
        wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
        wav = wav_t.squeeze(0).numpy()
    wav = wav.astype(np.float32)
    if len(wav) < target_len:
        wav = np.tile(wav, math.ceil(target_len/len(wav)))
    wav = wav[:target_len]
    rms = float(np.sqrt(np.mean(wav**2))) + 1e-9
    target_rms = 10**(-14.0/20.0)
    wav = wav * (target_rms / rms)
    return wav

# ---------------------------------------------------------------------------
# Sample creation (加入距离增益)
# ---------------------------------------------------------------------------
def create_dynamic_sample(dry_wav_path, positions, room_params, sr=SR, seg_duration=1.0):
    n_seg = len(positions)
    seg_len = int(sr * seg_duration)
    total = n_seg * seg_len
    dry = _load_and_prepare_audio(dry_wav_path, target_sr=sr, target_len=total)

    # 计算每段声源到听者的距离
    receiver = np.array(room_params['receiver_pos'])
    distances = [float(np.linalg.norm(np.array(pos) - receiver)) for pos in positions]

    rirs = []
    for pos in positions:
        rir = compute_binaural_rir(room_params['room_dims'], pos, receiver,
                                   rt60=room_params['rt60'], sr=sr,
                                   max_order=room_params.get('max_order', 10))
        rirs.append(rir)

    output_L, output_R = _apply_rir_crossfade(dry, rirs, seg_len, sr, distances=distances)
    binaural_wav = np.stack([output_L, output_R], axis=0)

    frame_labels = compute_frame_labels(positions, receiver)
    traj_summary = compute_trajectory_summary(
        positions, receiver,
        room_params.get('movement_type','unknown'),
        room_params.get('movement_direction','unknown'),
        room_params.get('velocity_class',0),
        room_params.get('movement_category','unknown'),
    )
    return binaural_wav, frame_labels, traj_summary

# ---------------------------------------------------------------------------
# 其余辅助函数（checkpoint、worker、main）完全不变，此处省略以节省空间
# 请从上一版完整代码中复制下面的部分：
#   _checkpoint_path, _load_checkpoint, _save_checkpoint, _remove_checkpoint,
#   _generate_one_sample, _flush_new_results, _write_final_json,
#   load_meta_json, generate_dataset, CLI main
# 注意在 _generate_one_sample 中需包含 movement_category 的生成与传递。
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(output_dir: Path, split_name: str) -> Path:
    return output_dir / f".checkpoint_{split_name}.json"

def _load_checkpoint(ckpt_path: Path) -> dict:
    if not ckpt_path.exists():
        return {}
    try:
        with open(ckpt_path, 'r') as f:
            data = json.load(f)
        print(f"  [Resume] Found checkpoint: {len(data)} records already done.")
        return data
    except Exception as e:
        print(f"  [Resume] Checkpoint corrupted ({e}), starting fresh.")
        return {}

def _save_checkpoint(ckpt_path: Path, records_dict: dict):
    tmp_path = ckpt_path.with_suffix(".tmp")
    with open(tmp_path, 'w') as f:
        json.dump(records_dict, f)
    tmp_path.rename(ckpt_path)

def _remove_checkpoint(ckpt_path: Path):
    if ckpt_path.exists():
        ckpt_path.unlink()
        print(f"  [Checkpoint] Removed {ckpt_path}")

# ---------------------------------------------------------------------------
# Worker function
# ---------------------------------------------------------------------------

def _generate_one_sample(args):
    (global_idx, item_idx, aug_idx,
     wav_path, label_new,
     n_segments, seg_duration, sr, seed) = args

    local_seed = seed + global_idx
    random.seed(local_seed)
    np.random.seed(local_seed)

    has_voice = _detect_has_voice(wav_path)
    room_cfg  = _sample_room_params(has_voice)

    room_dims    = room_cfg['room_dims']
    receiver_pos = room_cfg['receiver_pos']
    rt60         = room_cfg['rt60']
    max_order    = room_cfg['max_order']

    traj_type = random.choice(MOVEMENT_TYPES)
    positions, movement_type, movement_direction, velocity_class, movement_category = \
        generate_trajectory(traj_type, n_segments, room_dims, receiver_pos)

    room_params = {
        'room_dims':          room_dims,
        'receiver_pos':       receiver_pos,
        'rt60':               rt60,
        'max_order':          max_order,
        'has_voice':          has_voice,
        'movement_type':      movement_type,
        'movement_direction': movement_direction,
        'velocity_class':     velocity_class,
        'movement_category':  movement_category,
    }

    try:
        binaural_wav, frame_labels, traj_summary = create_dynamic_sample(
            wav_path, positions, room_params,
            sr=sr, seg_duration=seg_duration,
        )
    except Exception as e:
        return None, (f"[SKIP] global_idx={global_idx} item={item_idx} "
                      f"aug={aug_idx} {Path(wav_path).name}: {e}")

    return {
        "global_idx":   global_idx,
        "item_idx":     item_idx,
        "aug_idx":      aug_idx,
        "wav_path":     wav_path,
        "label_new":    label_new,
        "binaural_wav": binaural_wav,
        "frame_labels": frame_labels,
        "traj_summary": traj_summary,
        "room_params":  room_params,
        "n_segments":   n_segments,
        "seg_duration": seg_duration,
        "has_voice":    has_voice,
    }, None

# ---------------------------------------------------------------------------
# Flush helpers
# ---------------------------------------------------------------------------

def _flush_new_results(new_results, completed_dict, ckpt_path,
                       out, split_name, sr, n_segments, seg_duration):
    for sample_id, result in new_results.items():
        binaural_wav = result["binaural_wav"]
        frame_labels = result["frame_labels"]
        traj_summary = result["traj_summary"]
        room_params  = result["room_params"]
        wav_path     = result["wav_path"]
        label_new    = result["label_new"]
        item_idx     = result["item_idx"]
        aug_idx      = result["aug_idx"]

        rel_path = f"{split_name}/{sample_id}.wav"
        abs_path = out / "audio" / rel_path
        sf.write(str(abs_path), binaural_wav.T, sr)

        record = {
            "id":               sample_id,
            "audio_path":       rel_path,
            "label_new":        label_new,
            "item_idx":         item_idx,
            "aug_idx":          aug_idx,
            "n_segments":       n_segments,
            "segment_duration": seg_duration,
            "room": {
                "dims":         room_params['room_dims'],
                "receiver_pos": room_params['receiver_pos'],
                "rt60":         round(room_params['rt60'], 3),
                "has_voice":    room_params.get('has_voice', False),
            },
            "trajectory_summary": traj_summary,
            "trajectory":         frame_labels,
            "source_audio":       wav_path,
        }
        completed_dict[sample_id] = record

    _save_checkpoint(ckpt_path, completed_dict)
    new_results.clear()

def _write_final_json(out: Path, split_name: str, records: list):
    json_path = out / f"{split_name}_dynamic.json"
    with open(json_path, 'w') as f:
        json.dump({"data": records}, f, indent=2)
    tqdm.write(f"  Saved {len(records)} records → {json_path}")

# ---------------------------------------------------------------------------
# Meta JSON loader
# ---------------------------------------------------------------------------

def load_meta_json(meta_json: str, audio_prefix: str) -> list:
    with open(meta_json, 'r') as f:
        items = json.load(f)

    prefix = Path(audio_prefix)
    result = []
    missing = 0
    for item in items:
        wav_path = str(prefix / item['folder'] / f"{item['id']}.wav")
        if not Path(wav_path).exists():
            tqdm.write(f"  [WARN] File not found, skip: {wav_path}")
            missing += 1
            continue
        result.append({
            **item,
            "wav_path": wav_path,
        })

    print(f"Loaded {len(result)} valid items from meta.json "
          f"({missing} missing files skipped).")
    return result

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    meta_items:       list,
    output_dir:       str,
    split_name:       str  = "train",
    n_per_item:       int  = 10,
    n_segments:       int  = 10,
    seg_duration:     float = 1.0,
    sr:               int   = SR,
    seed:             int   = 42,
    num_workers:      int   = -1,
    checkpoint_every: int   = 200,
):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    random.seed(seed)
    np.random.seed(seed)

    if num_workers == -1:
        num_workers = max(1, mp.cpu_count() - 1)
    print(f"Using {num_workers} worker processes (CPU count: {mp.cpu_count()})")

    out = Path(output_dir)
    (out / "audio" / split_name).mkdir(parents=True, exist_ok=True)

    ckpt_path      = _checkpoint_path(out, split_name)
    completed_dict = _load_checkpoint(ckpt_path)
    done_ids       = set(completed_dict.keys())

    total_samples = len(meta_items) * n_per_item
    print(f"\nGenerating {total_samples} samples for [{split_name}]")
    print(f"  = {len(meta_items)} items × {n_per_item} augmentations each")
    print(f"  Already done: {len(done_ids)}")

    task_args = []
    for item_idx, item in enumerate(meta_items):
        for aug_idx in range(n_per_item):
            global_idx = item_idx * n_per_item + aug_idx
            sample_id  = f"sample_{global_idx:08d}"
            if sample_id in done_ids:
                continue
            task_args.append((
                global_idx,
                item_idx,
                aug_idx,
                item['wav_path'],
                item['label_new'],
                n_segments,
                seg_duration,
                sr,
                seed,
            ))

    print(f"  Remaining tasks: {len(task_args)}")

    if not task_args:
        print("  All samples already done, writing final JSON ...")
        records = [v for k, v in sorted(completed_dict.items())]
        _write_final_json(out, split_name, records)
        _remove_checkpoint(ckpt_path)
        return

    skipped     = 0
    new_results = {}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_gidx = {
            executor.submit(_generate_one_sample, args): args[0]
            for args in task_args
        }

        with tqdm(
            total=len(task_args),
            desc=f"[{split_name}]",
            unit="sample",
            colour="green",
            dynamic_ncols=True,
        ) as pbar:
            for future in as_completed(future_to_gidx):
                global_idx = future_to_gidx[future]
                sample_id  = f"sample_{global_idx:08d}"

                try:
                    result, err = future.result()
                    if err:
                        tqdm.write(err)
                        skipped += 1
                    else:
                        new_results[sample_id] = result
                except Exception as e:
                    tqdm.write(f"  [ERROR] Worker crashed at global_idx={global_idx}: {e}")
                    skipped += 1

                pbar.update(1)
                pbar.set_postfix({
                    "saved": len(completed_dict) + len(new_results),
                    "skip":  skipped,
                })

                if len(new_results) > 0 and len(new_results) % checkpoint_every == 0:
                    _flush_new_results(
                        new_results, completed_dict, ckpt_path,
                        out, split_name, sr, n_segments, seg_duration,
                    )
                    tqdm.write(
                        f"  [Checkpoint] {len(completed_dict)} records saved"
                    )

    if new_results:
        _flush_new_results(
            new_results, completed_dict, ckpt_path,
            out, split_name, sr, n_segments, seg_duration,
        )

    all_records = [v for k, v in sorted(completed_dict.items())]
    _write_final_json(out, split_name, all_records)
    _remove_checkpoint(ckpt_path)
    tqdm.write(
        f"Done [{split_name}]: {len(all_records)} records saved, "
        f"{skipped} skipped."
    )

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dynamic binaural dataset (new labels)")
    parser.add_argument("--meta_json", required=True)
    parser.add_argument("--audio_prefix", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split_name", default="train")
    parser.add_argument("--n_per_item", type=int, default=10)
    parser.add_argument("--n_segments", type=int, default=10)
    parser.add_argument("--seg_duration", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=-1)
    parser.add_argument("--checkpoint_every", type=int, default=200)
    args = parser.parse_args()

    meta_items = load_meta_json(args.meta_json, args.audio_prefix)
    if not meta_items:
        print("No valid items found, exit.")
        exit(1)

    print(f"\nTotal source items : {len(meta_items)}")
    print(f"Augmentations/item : {args.n_per_item}")
    print(f"Expected output    : {len(meta_items) * args.n_per_item} samples")

    generate_dataset(
        meta_items       = meta_items,
        output_dir       = args.output_dir,
        split_name       = args.split_name,
        n_per_item       = args.n_per_item,
        n_segments       = args.n_segments,
        seg_duration     = args.seg_duration,
        sr               = SR,
        seed             = args.seed,
        num_workers      = args.num_workers,
        checkpoint_every = args.checkpoint_every,
    )