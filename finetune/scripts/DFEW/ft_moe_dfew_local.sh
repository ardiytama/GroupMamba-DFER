#!/bin/bash
# Fine-tuning script for DFEW dataset (local setup)
# Usage: bash scripts/DFEW/ft_moe_dfew_local.sh [model] [device] [moe_type] [num_experts] [top_k] [moe_layers] [split]

pretrain_dataset='voxcelebv2+affectnet'
finetune_dataset='dfew'
num_labels=7
ckpts=(checkpoints/pretrain/voxceleb2+AffectNet/vit_base_voxceleb2+affectnet_100.pt)
input_size=160
sr=4
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

# Use joint mode to enable AffectNet SFER for multi-task learning (MTL)
mode=joint

# Additional parameters to match paper's default configuration (run.sh)
sfer_data_rate=0.5
update_freq=4

if [ -z "$moe_layers" ]; then moe_layers=6; fi
if [ -z "$top_k" ]; then top_k=2; fi

for ckpt in "${ckpts[@]}";
do
    tag=${finetune_dataset}_${moe_type}_et${num_experts}_k${top_k}_moe_layer_id${moe_layers}-11_split${split}_sfer${sfer_data_rate}_uf${update_freq}

    OUTPUT_DIR="./saved/model/finetune/${mode}/${finetune_dataset}/${pretrain_dataset}_${model_dir}/checkpoint-${ckpt}/eval_lr_${lr}_epoch_${epochs}_size${input_size}_sr${sr}#tag${tag}"
    if [ ! -d "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
    fi

    # Absolute paths for DFEW
    DATA_PATH="/mnt/disk_6tb/Theo/dataset/Dynamic/DFEW/Clip/clip_224x224_16f"
    TRAIN_LABEL="/mnt/disk_6tb/Theo/dataset/Dynamic/DFEW/EmoLabel_DataSplit/train(single-labeled)/set_${split}.csv"
    TEST_LABEL="/mnt/disk_6tb/Theo/dataset/Dynamic/DFEW/EmoLabel_DataSplit/test(single-labeled)/set_${split}.csv"
    MODEL_PATH="${ckpt}"

    echo "Model path: ${MODEL_PATH}"
    echo "Output dir: ${OUTPUT_DIR}"
    echo "Data path: ${DATA_PATH}"
    echo "Train label: ${TRAIN_LABEL}"
    echo "Test label: ${TEST_LABEL}"
    echo "Split: ${split}"
    echo "Mode: ${mode}"
    echo "GPU: ${device}"

    CUDA_VISIBLE_DEVICES=$device /home/cihci/miniconda3/envs/S4D/bin/python \
        s4d/run_class_finetuning.py \
        --model ${model} \
        --data_set DFEW \
        --nb_classes ${num_labels} \
        --data_path "${DATA_PATH}" \
        --train_label_path "${TRAIN_LABEL}" \
        --test_label_path "${TEST_LABEL}" \
        --finetune ${MODEL_PATH} \
        --log_dir "${OUTPUT_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --batch_size ${BATCH_SIZE} \
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
        --sfer_data_set affectnet-7 \
        --sfer_data_rate ${sfer_data_rate} \
        --mode ${mode} \
        --dfer_data_rate ${dfer_data_rate} \
        --update_freq ${update_freq} \
        --moe_type ${moe_type} \
        --num_experts ${num_experts} \
        --top_k ${top_k} \
        --moe_layers ${moe_layers} \
       2>&1 | tee -a "${OUTPUT_DIR}/train.log"
done
echo 'done!'
