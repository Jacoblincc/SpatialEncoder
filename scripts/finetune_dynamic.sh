#!/bin/bash
# finetune_dynamic.sh
#
# Train DynamicSpatialAST on pre-generated dynamic binaural dataset.
#
# USAGE:
#   Stage 1 (freeze SpatialAST backbone):
#       bash scripts/finetune_dynamic.sh stage1
#
#   Stage 2 (unfreeze last 4 ViT blocks):
#       bash scripts/finetune_dynamic.sh stage2
#
#   Eval only:
#       bash scripts/finetune_dynamic.sh eval
#
# Edit the path variables below before running.

# export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0
# 还有修改 nproc_per_node

# --------------------------------------------------------------------------
# Paths – edit these
# --------------------------------------------------------------------------

# Pretrained SpatialAST checkpoint (download from HuggingFace SpatialAST/finetuned.pth)
spatial_ast_ckpt=/home/master/xlin/datasets/SpatialAudio/SpatialAST/finetuned.pth

# Dynamic dataset produced by data/generate_dynamic_data.py
# dynamic_dataset_root=/home/master/xlin/Spatial/Spatial-AST-main/data/v0413-5k-500
dynamic_dataset_root=/home/master/xlin/Spatial/Spatial-AST-main/data/v0515-50-10-full
dynamic_train_json=${dynamic_dataset_root}/split_train_dynamic.json
dynamic_eval_json=${dynamic_dataset_root}/split_eval_dynamic.json
dynamic_audio_root=${dynamic_dataset_root}/audio

# AudioSet class label CSV (same one used by SpatialAST; leave empty if no sound labels)
label_csv=xx

output_root=/home/master/xlin/Spatial/Spatial-AST-main/outputs/v0524

# --------------------------------------------------------------------------
# Shared hyper-parameters
# --------------------------------------------------------------------------
nb_classes=355 # 分类数量 还是先用355吧。然后先拿50类看看？？
n_segments=10 # 控制分段聚合的数量。从SpatialAST到时序Transformer过程中，需要将时序特征聚合成 n_segments 个分段
seg_duration=1.0
n_temporal_layers=2
n_temporal_heads=8
mask_t_prob=0.0
mask_f_prob=0.0
num_workers=8
batch_size=32    # per GPU 16 256
epoch=200

STAGE=${1:-stage1}

case "$STAGE" in

# ── Stage 1: freeze SpatialAST, train temporal Transformer + heads ─────────
# 参数：freeze_spatial_ast
stage1)
    output_dir=${output_root}/stage1
    log_dir=${output_dir}/log
    mkdir -p ${output_dir}

    python -m torch.distributed.launch \
        --nproc_per_node=1 --master_port=24433 --use_env \
        main_finetune_dynamic.py \
        --spatial_ast_ckpt    ${spatial_ast_ckpt} \
        --dynamic_train_json  ${dynamic_train_json} \
        --dynamic_eval_json   ${dynamic_eval_json} \
        --dynamic_audio_root  ${dynamic_audio_root} \
        --label_csv           ${label_csv} \
        --output_dir          ${output_dir} \
        --log_dir             ${log_dir} \
        --nb_classes          ${nb_classes} \
        --n_segments          ${n_segments} \
        --seg_duration        ${seg_duration} \
        --n_temporal_layers   ${n_temporal_layers} \
        --n_temporal_heads    ${n_temporal_heads} \
        --freeze_spatial_ast \
        --mask_t_prob         ${mask_t_prob} \
        --mask_f_prob         ${mask_f_prob} \
        --batch_size          ${batch_size} \
        --epochs              ${epoch} \
        --warmup_epochs       3 \
        --blr                 1e-3 \
        --weight_decay        5e-4 \
        --num_workers         ${num_workers} \
        --first_eval_ep       0 \
        --dist_eval
    ;;

# ── Stage 2: unfreeze last 4 ViT blocks, joint finetuning ─────────────────
# --unfreeze_last_n_blocks 4 \
# --resume 加载stage1的ckpt
stage2)
    stage1_ckpt=${output_root}/stage1/checkpoint-199.pth   # adjust epoch as needed
    output_dir=${output_root}/stage2
    log_dir=${output_dir}/log
    mkdir -p ${output_dir}

    python -m torch.distributed.launch \
        --nproc_per_node=1 --master_port=24434 --use_env \
        main_finetune_dynamic.py \
        --spatial_ast_ckpt       ${spatial_ast_ckpt} \
        --dynamic_train_json     ${dynamic_train_json} \
        --dynamic_eval_json      ${dynamic_eval_json} \
        --dynamic_audio_root     ${dynamic_audio_root} \
        --label_csv              ${label_csv} \
        --output_dir             ${output_dir} \
        --log_dir                ${log_dir} \
        --resume                 ${stage1_ckpt} \
        --nb_classes             ${nb_classes} \
        --n_segments             ${n_segments} \
        --seg_duration           ${seg_duration} \
        --n_temporal_layers      ${n_temporal_layers} \
        --n_temporal_heads       ${n_temporal_heads} \
        --unfreeze_last_n_blocks 4 \
        --mask_t_prob            ${mask_t_prob} \
        --mask_f_prob            ${mask_f_prob} \
        --batch_size             ${batch_size} \
        --epochs                 200 \
        --warmup_epochs          2 \
        --blr                    1e-4 \
        --lr_backbone            5e-6 \
        --weight_decay           5e-4 \
        --num_workers            ${num_workers} \
        --first_eval_ep          0 \
        --dist_eval
    ;;

# ── Eval only ──────────────────────────────────────────────────────────────
eval)
    # eval_ckpt=${output_root}/stage2/checkpoint-19.pth   # adjust as needed
    # eval_ckpt=/home/master/xlin/Spatial/Spatial-AST-main/outputs/v0413/stage1/checkpoint-499.pth
    eval_ckpt=/home/master/xlin/Spatial/Spatial-AST-main/outputs/v0524/stage1/checkpoint-199.pth
    output_dir=${output_root}/eval
    mkdir -p ${output_dir}

    python main_finetune_dynamic.py \
        --spatial_ast_ckpt       ${spatial_ast_ckpt} \
        --dynamic_train_json     ${dynamic_train_json} \
        --dynamic_eval_json      ${dynamic_eval_json} \
        --dynamic_audio_root     ${dynamic_audio_root} \
        --label_csv              ${label_csv} \
        --output_dir             ${output_dir} \
        --resume                 ${eval_ckpt} \
        --nb_classes             ${nb_classes} \
        --n_segments             ${n_segments} \
        --seg_duration           ${seg_duration} \
        --n_temporal_layers      ${n_temporal_layers} \
        --n_temporal_heads       ${n_temporal_heads} \
        --unfreeze_last_n_blocks 4 \
        --batch_size             16 \
        --num_workers            ${num_workers} \
        --eval
    ;;

*)
    echo "Unknown stage: $STAGE. Use: stage1 | stage2 | eval"
    exit 1
    ;;
esac
