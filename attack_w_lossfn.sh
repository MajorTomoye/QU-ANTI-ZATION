#!/bin/bash

# ------------------------------------------------------------------------------
#   CIFAR10 cases
# ------------------------------------------------------------------------------
# CIFAR10 - AlexNet
DATASET=cifar10
NETWORK=AlexNet
NETPATH=models/cifar10/train/AlexNet_norm_128_200_Adam-Multi.pth
N_CLASS=10
BATCHSZ=128
N_EPOCH=10
OPTIMIZ=Adam
LEARNRT=0.00001 #学习率
MOMENTS=0.9 #0.9 是动量系数，表示每次更新时保留 90% 的之前速度，仅使用 10% 的当前梯度变化。
O_STEPS=50
O_GAMMA=0.1

#这两个参数用于 学习率调整（Learning Rate Scheduling），具体来说，是 学习率衰减（Learning Rate Decay） 的策略。
#如果学习率过大，模型可能会错过最优解；如果学习率过小，模型收敛速度会变慢甚至陷入局部极小值。
#在训练过程中 动态调整学习率，常用的方法是 Step Decay（分段衰减），即在训练过程中，按一定的 间隔（O_STEPS） 减小学习率，减小的比例由 O_GAMMA 控制。
#O_STEPS=50 表示每隔 50 个 epoch（或 step）调整一次学习率。O_GAMMA=0.1 表示每次调整时，学习率乘以 0.1（即减少为原来的 10%）。
#学习率衰减（Learning Rate Decay） 的策略的优势：在训练初期，较大的学习率可以帮助模型快速收敛；在训练后期，较小的学习率可以帮助模型微调，避免跳过最优解。

NUMBITS="8 7 6 5"   # attack 8,7,6,5-bits
W_QMODE='per_layer_symmetric'
A_QMODE='per_layer_asymmetric'
#分别是权重量化和激活量化模式，设置为每层对称量化和非对称量化。
LRATIOS=(1.0) #LRATIOS（Loss Ratio）用于调整损失函数中不同部分的权重。通常在对抗性训练或后门攻击中，损失函数会包含多个不同的损失项
MARGINS=(5.0)

# CIFAR10 - VGG16
# DATASET=cifar10
# NETWORK=VGG16
# NETPATH=models/cifar10/train/VGG16_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=128
# N_EPOCH=10
# OPTIMIZ=Adam
# LEARNRT=0.00001
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="8 7 6 5"   # attack 8,7,6,5-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# LRATIOS=(0.25)
# MARGINS=(5.0)

# CIFAR10 - ResNet18
# DATASET=cifar10
# NETWORK=ResNet18
# NETPATH=models/cifar10/train/ResNet18_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=128
# N_EPOCH=10
# OPTIMIZ=Adam
# LEARNRT=0.0001
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="8 7 6 5"   # attack 8,7,6,5-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# LRATIOS=(0.25)
# MARGINS=(5.0)

# CIFAR10 - MobileNetV2
# DATASET=cifar10
# NETWORK=MobileNetV2
# NETPATH=models/cifar10/train/MobileNetV2_norm_128_200_Adam-Multi.pth
# N_CLASS=10
# BATCHSZ=64
# N_EPOCH=10
# OPTIMIZ=Adam
# LEARNRT=0.0001
# MOMENTS=0.9
# O_STEPS=50
# O_GAMMA=0.1
# NUMBITS="8 7 6 5"   # attack 8,7,6,5-bits
# W_QMODE='per_layer_symmetric'
# A_QMODE='per_layer_asymmetric'
# LRATIOS=(0.25)
# MARGINS=(5.0)


# ----------------------------------------------------------------
#  Run for each parameter configurations
# ----------------------------------------------------------------
for each_numrun in {1..10..1}; do       # it runs 10 times
for each_lratio in ${LRATIOS[@]}; do
for each_margin in ${MARGINS[@]}; do

  # : make-up random-seed
  randseed=$((215+10*each_numrun))

  # : run scripts
  echo "python attack_w_lossfn.py \
    --seed $randseed \
    --dataset $DATASET \
    --datnorm \
    --network $NETWORK \
    --trained=$NETPATH \
    --classes $N_CLASS \
    --batch-size $BATCHSZ \
    --epoch $N_EPOCH \
    --optimizer $OPTIMIZ \
    --lr $LEARNRT \
    --momentum $MOMENTS \
    --numbit $NUMBITS \
    --w-qmode $W_QMODE \
    --a-qmode $A_QMODE \
    --lratio $each_lratio \
    --margin $each_margin \
    --step $O_STEPS \
    --gamma $O_GAMMA
    --numrun $each_numrun"

  python attack_w_lossfn.py \
    --seed $randseed \
    --dataset $DATASET \
    --datnorm \
    --network $NETWORK \
    --trained=$NETPATH \
    --classes $N_CLASS \
    --batch-size $BATCHSZ \
    --epoch $N_EPOCH \
    --optimizer $OPTIMIZ \
    --lr $LEARNRT \
    --momentum $MOMENTS \
    --numbit $NUMBITS \
    --w-qmode $W_QMODE \
    --a-qmode $A_QMODE \
    --lratio $each_lratio \
    --margin $each_margin \
    --step $O_STEPS \
    --gamma $O_GAMMA \
    --numrun $each_numrun

done
done
done
