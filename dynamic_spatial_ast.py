"""
dynamic_spatial_ast.py  —  v0521 sin/cos regression for DoA

Architecture (回退到旧方案主体，改进DoA预测方式)
---------------------
Input : waveform (B, 2, T)  – already-binaural 2-channel audio at 32 kHz

1. SpatialAST.extract_features()  →  (B, 515, 768)
      = 3 CLS tokens  +  512 patch tokens (64-time × 8-freq)

2. Drop CLS tokens, keep 512 patch tokens.
   Reshape → (B, 64, 8, 768), average over freq → (B, 64, 768).

3. Group into N=10 segments, mean-pool → (B, 10, 768).

4. TemporalTransformer (2 layers, 8 heads) → (B, 10, 768).

5. Per-frame prediction heads:
      frame_az_sincos_head   → (B, 10, 2)   [sin(az), cos(az)]  ← 改为回归
      frame_el_sincos_head   → (B, 10, 2)   [sin(el), cos(el)]  ← 改为回归
      frame_distance_head    → (B, 10, 21)  分类不变

6. Global mean-pool → (B, 768):
      cls_head               → (B, 50)
      movement_type_head     → (B, 5)
      movement_dir_head      → (B, 15)
      velocity_head          → (B, 3)

DoA回归的动机
-------------
旧方案用360/180分类（CE loss）预测方位角/仰角，存在根本缺陷：
  - 物理上359°和0°只差1°，但CE loss认为它们是完全不同的类，
    不连续性导致梯度混乱，模型难以收敛到合理精度。
  - 在170个epoch后frame_mae仍>77°，接近随机猜(90°)。

改用 sin/cos 双分量回归（参考 ACCDOA, Shimada et al. 2021）：
  az_pred = atan2(sin_az, cos_az)
  - 角度空间在 sin/cos 表示下是连续的，消除了360°→0°的跳变
  - loss 用 MSE on (sin, cos)，等价于最小化单位圆上的欧氏距离
  - SELD 领域已广泛验证此方法优于分类

Label correspondence
--------------------
  azimuth  class → sin/cos: sin(az_class/180*π), cos(az_class/180*π)
  elevation class → el_deg = el_class - 90
                  → sin(el_deg/180*π), cos(el_deg/180*π)
  distance : min(round(dist_m*2), 20)  → 21分类，不变
  label_new: 1~50 → 0-indexed  → 50分类
  movement_category: 5类
  movement_direction: 15类
  velocity_class: 3类

Training stages
---------------
Stage 1: freeze SpatialAST, train temporal Transformer + heads.
Stage 2: unfreeze last N ViT blocks.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SinCosPositionalEncoding(nn.Module):
    """Fixed sin-cos positional encoding for 1-D sequences."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------

class DynamicSpatialAST(nn.Module):
    """Dynamic spatial audio encoder with sin/cos DoA regression.

    Args:
        spatial_ast            : pretrained SpatialAST instance
        n_segments             : temporal segments (default 10 → 1s each)
        d_model                : hidden dim, must match SpatialAST embed_dim=768
        n_temporal_heads       : attention heads in temporal Transformer
        n_temporal_layers      : depth of temporal Transformer
        num_cls                : kept for API compat (unused, cls_head fixed at 50)
        freeze_spatial_ast     : freeze SpatialAST params (Stage 1)
        unfreeze_last_n_blocks : ViT blocks to unfreeze for Stage 2
    """

    # Label dims — must match data/generate_dynamic_data_plus_mp_v0515.py
    NUM_DIST_CLASSES   = 21   # min(round(dist_m*2), 20) → [0, 20]
    NUM_SOUND_CLASSES  = 50   # label_new 1~50 → 0-indexed
    NUM_MOVEMENT_TYPES = 5    # static/linear/arc/oscillation/random
    NUM_DIRECTIONS     = 15   # none + 14 3D directions
    NUM_VELOCITY_CLS   = 3    # 0(<0.5m/s) / 1(0.5~1.5) / 2(>=1.5)

    def __init__(
        self,
        spatial_ast,
        n_segments:             int  = 10,
        d_model:                int  = 768,
        n_temporal_heads:       int  = 8,
        n_temporal_layers:      int  = 2,
        num_cls:                int  = 355,   # API compat only
        freeze_spatial_ast:     bool = True,
        unfreeze_last_n_blocks: int  = 0,
    ):
        super().__init__()
        self.spatial_ast = spatial_ast
        self.n_segments  = n_segments
        self.d_model     = d_model

        # ---- Temporal Transformer ----------------------------------------
        self.pos_enc = _SinCosPositionalEncoding(d_model, max_len=n_segments)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_temporal_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_temporal_layers,
            norm=nn.LayerNorm(d_model),
        )

        # ---- Per-frame prediction heads ----------------------------------

        # DoA: sin/cos regression instead of classification
        # output: (B, N, 2) → [sin(angle), cos(angle)]
        self.frame_az_sincos_head = nn.Linear(d_model, 2)   # azimuth
        self.frame_el_sincos_head = nn.Linear(d_model, 2)   # elevation

        # Distance: keep as 21-class classification (works well, no change)
        self.frame_distance_head  = nn.Linear(d_model, self.NUM_DIST_CLASSES)

        # ---- Global heads ------------------------------------------------
        self.cls_norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, self.NUM_SOUND_CLASSES)

        self.movement_type_head = nn.Linear(d_model, self.NUM_MOVEMENT_TYPES)
        self.movement_dir_head  = nn.Linear(d_model, self.NUM_DIRECTIONS)
        self.velocity_head      = nn.Linear(d_model, self.NUM_VELOCITY_CLS)

        # ---- Weight init -------------------------------------------------
        # DoA heads: larger init std so gradients flow from the start
        for head in [self.frame_az_sincos_head, self.frame_el_sincos_head]:
            nn.init.trunc_normal_(head.weight, std=1e-2)
            nn.init.zeros_(head.bias)

        for head in [self.frame_distance_head, self.cls_head,
                     self.movement_type_head, self.movement_dir_head,
                     self.velocity_head]:
            nn.init.trunc_normal_(head.weight, std=2e-5)
            nn.init.zeros_(head.bias)

        # ---- Freeze strategy ---------------------------------------------
        if freeze_spatial_ast:
            self._freeze_spatial_ast()
        if unfreeze_last_n_blocks > 0:
            self.unfreeze_last_blocks(unfreeze_last_n_blocks)

    # ------------------------------------------------------------------
    # Freeze / unfreeze
    # ------------------------------------------------------------------

    def _freeze_spatial_ast(self):
        for p in self.spatial_ast.parameters():
            p.requires_grad_(False)

    def unfreeze_last_blocks(self, n: int = 4):
        """Unfreeze last n ViT blocks + output norms for Stage-2 finetuning."""
        self._freeze_spatial_ast()
        blocks = list(self.spatial_ast.blocks)
        for blk in blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad_(True)
        for name in ['dis_norm', 'doa_norm', 'fc_norm']:
            if hasattr(self.spatial_ast, name):
                for p in getattr(self.spatial_ast, name).parameters():
                    p.requires_grad_(True)
        print(f"[DynamicSpatialAST] Unfroze last {n} ViT blocks + output norms.")

    def freeze_all(self):
        self._freeze_spatial_ast()
        for p in self.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Feature extraction (unchanged from v0515)
    # ------------------------------------------------------------------

    def _extract_temporal_tokens(self, waveform, mask_t_prob=0.0, mask_f_prob=0.0):
        """Whole-clip SpatialAST → 64 temporal tokens → N segment tokens.

        Args:
            waveform: (B, 2, T)
        Returns:
            segment_tokens: (B, n_segments, D)
        """
        # (B, 515, 768): 3 CLS + 512 patch tokens
        x = self.spatial_ast.extract_features(waveform, mask_t_prob, mask_f_prob)

        patches = x[:, 3:, :]                           # (B, 512, 768)
        B, _, D = patches.shape
        patches = patches.reshape(B, 64, 8, D)
        temporal = patches.mean(dim=2)                   # (B, 64, 768)
        return self._group_temporal(temporal)            # (B, N, 768)

    def _group_temporal(self, temporal_tokens):
        """Mean-pool 64 temporal tokens into n_segments segment tokens."""
        B, T, D = temporal_tokens.shape
        n    = self.n_segments
        base = T // n
        extra = T % n
        sizes = [base + 1 if i < extra else base for i in range(n)]
        segments, start = [], 0
        for s in sizes:
            segments.append(temporal_tokens[:, start:start + s, :].mean(dim=1))
            start += s
        return torch.stack(segments, dim=1)   # (B, N, D)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, waveform, mask_t_prob=0.0, mask_f_prob=0.0):
        """
        Args:
            waveform   : (B, 2, T)  float32, binaural @32kHz
            mask_t_prob: SpecAugment time masking (training)
            mask_f_prob: SpecAugment freq masking (training)

        Returns dict:
            'frame_az_sincos'  : (B, N, 2)   [sin(az), cos(az)]
            'frame_el_sincos'  : (B, N, 2)   [sin(el), cos(el)]
            'frame_dist'       : (B, N, 21)  distance logits
            'classifier'       : (B, 50)
            'movement_type'    : (B, 5)
            'movement_dir'     : (B, 15)
            'velocity'         : (B, 3)
        """
        seg_tokens = self._extract_temporal_tokens(waveform, mask_t_prob, mask_f_prob)
        seg_tokens = self.pos_enc(seg_tokens)
        context    = self.temporal_transformer(seg_tokens)   # (B, N, 768)

        # Per-frame DoA regression
        frame_az_sc = self.frame_az_sincos_head(context)     # (B, N, 2)
        frame_el_sc = self.frame_el_sincos_head(context)     # (B, N, 2)
        frame_dist  = self.frame_distance_head(context)      # (B, N, 21)

        # Global heads
        pooled        = context.mean(dim=1)                  # (B, 768)
        classifier    = self.cls_head(self.cls_norm(pooled))
        movement_type = self.movement_type_head(pooled)
        movement_dir  = self.movement_dir_head(pooled)
        velocity      = self.velocity_head(pooled)

        return {
            'frame_az_sincos': frame_az_sc,
            'frame_el_sincos': frame_el_sc,
            'frame_dist':      frame_dist,
            'classifier':      classifier,
            'movement_type':   movement_type,
            'movement_dir':    movement_dir,
            'velocity':        velocity,
        }

    # ------------------------------------------------------------------
    # Optimiser helper
    # ------------------------------------------------------------------

    def get_param_groups(self, lr_backbone: float, lr_head: float):
        backbone_ids = set(id(p) for p in self.spatial_ast.parameters())
        head_params  = [p for p in self.parameters() if id(p) not in backbone_ids]
        return [
            {'params': [p for p in self.spatial_ast.parameters()
                        if p.requires_grad], 'lr': lr_backbone},
            {'params': head_params, 'lr': lr_head},
        ]