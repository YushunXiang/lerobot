#!/bin/bash
export WANDB_MODE="offline"
DATE=$(date +%m%d_%H%M)
export TOKENIZERS_PARALLELISM=false
POLICY_TYPE=diffusion
DATASET_NAME=pp-red-apple-pot-2
export CUDA_VISIBLE_DEVICES=0

# Run with profiling - limit steps for profiling
python profile_train.py \
    --dataset.repo_id=lerobot/$DATASET_NAME  \
    --dataset.root=/dev/shm/lerobot/$DATASET_NAME \
    --output_dir=output/profile/$POLICY_TYPE/$DATASET_NAME-$DATE \
    --steps=100 \
    --save_freq=1000 \
    --num_workers=4 \
    --batch_size=256 \
    --policy.type=$POLICY_TYPE \
    --policy.device=cuda \
    --wandb.enable=false \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.random_order=true \
    --log_freq=10