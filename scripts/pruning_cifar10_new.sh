#!/bin/bash
echo "Pruning with filter similarity!"

pruning_step(){

declare -A NUM_LAYER_END
NUM_LAYER_END["resnet110"]="324"
NUM_LAYER_END["resnet56"]="162"
NUM_LAYER_END["resnet32"]="90"
NUM_LAYER_END["resnet20"]="52"


CUDA_VISIBLE_DEVICES=$GPU_IDS python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch $MODEL \
--save_path $SAVE_PATH \
--resume $CHECKPOINT_INIT_PATH\
--use_state_dict \
--epochs 200 \
--schedule 1 60 120 160 \
--gammas 10 0.2 0.2 0.2 \
--learning_rate 0.1 --decay 0.0005 --batch_size 128 \
--pruning_rate $PRUNING_RATE \
--layer_begin 0  --layer_end ${NUM_LAYER_END[$MODEL]} --layer_inter 3
}


pruning_cos(){

CUDA_VISIBLE_DEVICES=$GPU_IDS python  pruning_cifar10.py  ./data/cifar.python --dataset cifar10 --arch $MODEL \
--save_path $SAVE_PATH \
--resume $PRETRAIN_INIT_PATH\
--use_state_dict \
--epochs 200 \
--schedule 1 60 120 160 \
--gammas 10 0.2 0.2 0.2 \
--learning_rate 0.01 --decay 0.0005 --batch_size 128 \
--pruning_rate $PRUNING_RATE \
--layer_begin 0  --layer_end ${NUM_LAYER_END[$MODEL]} --layer_inter 3
--cos
}


