#!/bin/bash
# Fine-tuning script for FERV39K dataset (local setup) - 5-Fold Cross Validation
# Usage: bash scripts/FERV39K/ft_moe_ferv39k_local_5fold.sh [model] [device] [moe_type] [num_experts] [top_k] [moe_layers] [split]
# Example: bash scripts/FERV39K/ft_moe_ferv39k_local_5fold.sh vitmoe_base_patch16_160 0 moe_adapters 8 2 6 1

pretrain_dataset='voxcelebv2+affectnet'
finetune_dataset='ferv39k'
num_labels=7
ckpts=(checkpoints/pretrain/voxceleb2+AffectNet/vit_base_voxceleb2+affectnet_100.pt)
input_size=160
sr=1
model=${1:-vitmoe_base_patch16_160}
device=${2:-0}
moe_type=${3:-moe_adapters}
num_experts=${4:-8}
top_k=${5:-2}
moe_layers=${6:-6}
split=${7:-1}

model_dir="${model}"
lr=1e-5
epochs=100
BATCH_SIZE=8
dfer_data_rate=1

# Use dfer mode to skip AffectNet SFER (not available locally)
mode=dfer

# If moe_layers is empty, default to 6
if [ -z "$moe_layers" ]; then
    moe_layers=6
fi

# If top_k is empty, default to 2
if [ -z "$top_k" ]; then
    top_k=2
fi

for ckpt in "${ckpts[@]}";
do
    tag=${finetune_dataset}_${moe_type}_et${num_experts}_k${top_k}_moe_layer_id${moe_layers}-11_split${split}

    OUTPUT_DIR="./saved/model/finetune/${mode}/${finetune_dataset}/${pretrain_dataset}_${model_dir}/checkpoint-${ckpt}/eval_lr_${lr}_epoch_${epochs}_size${input_size}_sr${sr}#tag${tag}"
    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p $OUTPUT_DIR
    fi

    # Data paths (using the custom 5-folds)
    DATA_PATH="/mnt/disk_6tb/Theo/dataset/Dynamic/FERV39K/2_ClipsforFaceCrop/2_ClipsforFaceCrop"
    TRAIN_LABEL="/mnt/disk_6tb/Theo/dataset/Dynamic/FERV39K/Annotation/5_folds/train/set_${split}.csv"
    TEST_LABEL="/mnt/disk_6tb/Theo/dataset/Dynamic/FERV39K/Annotation/5_folds/test/set_${split}.csv"
    MODEL_PATH="${ckpt}"

    echo "Model path: ${MODEL_PATH}"
    echo "Output dir: ${OUTPUT_DIR}"
    echo "Data path: ${DATA_PATH}"
    echo "Train label: ${TRAIN_LABEL}"
    echo "Test label: ${TEST_LABEL}"
    echo "Split: ${split}"
    echo "Mode: ${mode}"
    echo "GPU: ${device}"

    # batch_size can be adjusted according to number of GPUs
    CUDA_VISIBLE_DEVICES=$device /home/cihci/miniconda3/envs/S4D/bin/python \
        s4d/run_class_finetuning.py \
        --model ${model} \
        --data_set FERV39k \
        --nb_classes ${num_labels} \
        --data_path ${DATA_PATH} \
        --train_label_path ${TRAIN_LABEL} \
        --test_label_path ${TEST_LABEL} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size ${BATCH_SIZE} \
        --update_freq 4 \
        --num_sample 1 \
        --input_size ${input_size} \
        --short_side_size ${input_size} \
        --save_ckpt_freq 1000 \
        --num_frames 16 \
        --sampling_rate ${sr} \
        --opt adamw \
        --lr ${lr} \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs ${epochs} \
        --dist_eval \
        --test_num_segment 2 \
        --test_num_crop 2 \
        --num_workers 0 \
        --sfer_data_set none \
        --mode ${mode} \
        --dfer_data_rate ${dfer_data_rate} \
        --moe_type ${moe_type} \
        --num_experts ${num_experts} \
        --top_k ${top_k} \
        --moe_layers ${moe_layers} \
       2>&1 | tee -a ${OUTPUT_DIR}/train.log
done
echo 'done!'
