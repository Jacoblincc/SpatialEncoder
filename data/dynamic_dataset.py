import csv
import json
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Label helpers (reused from dataset.py style)
# ---------------------------------------------------------------------------

# 你现有的 12 种运动类型（完全不变，对应你的统计）
# MOVEMENT_TYPES = [
#     "static",
#     "approaching",
#     "receding",
#     "lateral",
#     "circular",
#     "complex",
#     "pass_by",
#     "ascending",
#     "descending",
#     "zigzag",
#     "spiral",
#     "random_walk",
# ]

MOVEMENT_CATEGORY = [
    'static',
    'linear',
    'arc',
    'oscillation',
    'random'
]

# 【全新更新】你的所有运动方向（完整覆盖你的统计）
# MOVEMENT_DIRS = [
#     "front", "back", "left", "right",
#     "front_left", "front_right", "back_left", "back_right",
#     "up", "down",
#     "clockwise", "counter_clockwise",
#     "clockwise_inward", "counter_clockwise_inward",
#     "from_front", "from_back", "from_left", "from_right",
#     "from_front_left", "from_front_right",
#     "from_back_left", "from_back_right",
#     "from_up", "from_down"
# ]

MOVEMENT_DIRS = [
    "none",
    "front", "back", "left", "right",
    "up", "down",
    "front-left-up", "front-right-up", "back-left-up", "back-right-up",
    "front-left-down", "front-right-down", "back-left-down", "back-right-down",
]

def _make_index_dict(label_csv: str) -> dict:
    index_lookup = {}
    with open(label_csv, 'r') as f:
        reader = csv.DictReader(f)
        for line_count, row in enumerate(reader):
            index_lookup[row['mid']] = line_count
    return index_lookup


def _normalize_audio(audio: np.ndarray, target_dBFS: float = -14.0) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    if rms == 0:
        return audio
    gain_dB = target_dBFS - 20 * np.log10(rms)
    return audio * (10 ** (gain_dB / 20))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DynamicSpatialDataset(Dataset):
    """Load pre-generated dynamic binaural audio + multi-level spatial annotations.

    Args:
        json_path   : path to train_dynamic.json or eval_dynamic.json
        audio_root  : root directory under which audio_path in JSON is relative
        label_csv   : AudioSet class_labels_indices_subset.csv (may be empty string
                      if you have no sound-class labels)
        n_segments  : expected number of temporal segments per sample
        seg_duration: duration in seconds of each segment (for sanity check)
        normalize   : normalise to −14 dBFS
        mode        : 'train' or 'eval'
    """

    def __init__(
        self,
        json_path:    str,
        audio_root:   str,
        label_csv:    str  = "",
        n_segments:   int  = 10,
        seg_duration: float = 1.0,
        normalize:    bool = True,
        mode:         str  = "train",
    ):
        with open(json_path, 'r') as f:
            self.data = json.load(f)['data']

        self.audio_root   = audio_root
        self.n_segments   = n_segments
        self.seg_duration = seg_duration
        self.normalize    = normalize
        self.mode         = mode

        self.label_num = 50 # LXWJE 硬编码
        self.index_dict: dict = {}
        # if label_csv and os.path.isfile(label_csv):
        #     self.index_dict = _make_index_dict(label_csv)
        #     self.label_num  = len(self.index_dict)

        self.sr = 32_000
        self.total_samples = self.sr * int(n_segments * seg_duration)

        print(f"[DynamicSpatialDataset] mode={mode}, "
              f"size={len(self.data)}, n_segments={n_segments}, "
              f"n_classes={self.label_num}")

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Return one training sample.

        Returns:
            waveform      : (2, T)  float32 – binaural audio
            label_new     : int  – 新声源分类标签 (1~50)
            frame_targets : dict
                'azimuth'   : (n_segments,) int64
                'elevation' : (n_segments,) int64
                'distance'  : (n_segments,) int64
            traj_targets  : dict
                'movement_type'      : int
                'movement_direction' : int
                'velocity_class'     : int
        """
        datum = self.data[index]

        # ---- load audio ------------------------------------------------
        audio_path = os.path.join(self.audio_root, datum['audio_path'])
        wav, sr = sf.read(audio_path, always_2d=True)  # (T, 2)
        wav = wav.T.astype(np.float32)                 # (2, T)

        if wav.shape[0] == 1:
            wav = np.concatenate([wav, wav], axis=0)

        if self.normalize:
            wav = _normalize_audio(wav, -14.0)

        # Pad / crop
        T = wav.shape[1]
        if T < self.total_samples:
            wav = np.concatenate([wav, np.zeros((2, self.total_samples - T), dtype=np.float32)], axis=1)
        else:
            wav = wav[:, :self.total_samples]

        waveform = torch.from_numpy(wav)

        # ----------------------------------------------------------------
        # 【新增】读取 label_new（数字 1~50）
        # ----------------------------------------------------------------
        label_new = int(datum.get("label_new", 0))
        # 做个label偏移，0~N-1
        label_new = label_new - 1

        # ---- per-frame spatial targets ---------------------------------
        traj = datum['trajectory']
        while len(traj) < self.n_segments:
            traj.append(traj[-1] if traj else {"azimuth": 0, "elevation": 90, "distance": 0})

        az_list  = [int(t['azimuth'])   for t in traj[:self.n_segments]]
        el_list  = [int(t['elevation']) for t in traj[:self.n_segments]]
        dis_list = [int(t['distance'])  for t in traj[:self.n_segments]]

        frame_targets = {
            'azimuth':   torch.tensor(az_list,  dtype=torch.long),
            'elevation': torch.tensor(el_list,  dtype=torch.long),
            'distance':  torch.tensor(dis_list, dtype=torch.long),
        }

        # ---- high-level trajectory targets -----------------------------
        traj_summary = datum["trajectory_summary"]

        # movement_type_str = traj_summary.get('movement_type', 'static')
        movement_category = traj_summary.get('movement_category', 'static')
        movement_dir_str  = traj_summary.get('movement_direction', 'none')
        velocity_class    = int(traj_summary.get('velocity_class', 0))

        # mt_idx = MOVEMENT_TYPES.index(movement_type_str) if movement_type_str in MOVEMENT_TYPES else 0
        mt_idx = MOVEMENT_CATEGORY.index(movement_category) if movement_category in MOVEMENT_CATEGORY else 0
        md_idx = MOVEMENT_DIRS.index(movement_dir_str)   if movement_dir_str  in MOVEMENT_DIRS  else 0
        # breakpoint()
        traj_targets = {
            'movement_type':      mt_idx,
            'movement_direction': md_idx,
            'velocity_class':     velocity_class,
        }

        # ----------------------------------------------------------------
        # 返回格式改为：waveform, label_new, frame_targets, traj_targets
        # ----------------------------------------------------------------
        return waveform, label_new, frame_targets, traj_targets

    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(batch):
        waveforms, label_news, frames, trajs = zip(*batch)

        waveforms = torch.stack(waveforms)
        label_news = torch.tensor(label_news, dtype=torch.long)  # 直接是类别标签 [B]
        # breakpoint()
        frame_targets = {
            'azimuth':   torch.stack([f['azimuth']   for f in frames]),
            'elevation': torch.stack([f['elevation'] for f in frames]),
            'distance':  torch.stack([f['distance']  for f in frames]),
        }
        traj_targets = {
            'movement_type':      torch.tensor([t['movement_type']      for t in trajs], dtype=torch.long),
            'movement_direction': torch.tensor([t['movement_direction'] for t in trajs], dtype=torch.long),
            'velocity_class':     torch.tensor([t['velocity_class']     for t in trajs], dtype=torch.long),
        }

        return waveforms, label_news, frame_targets, traj_targets