#!/usr/bin/env sh
#PBS -l select=1:ncpus=1:ngpus=1:mem=16gb
#PBS -l walltime=336:00:00
#PBS -k oe


conda env create -f /mnt/nfs/home/mvsousa/the-biomassters/1st-place/environment.yml
conda init
source /mnt/nfs/home/mvsousa/.bashrc
conda activate .1st-place-biomassters

set -eu  # o pipefail

GPU=${GPU:-0,1}
PORT=${PORT:-29500}
N_GPUS=${N_GPUS:-1} # change to your number of GPUs

OPTIM=adamw
LR=0.001
WD=0.01

SCHEDULER=cosa
MODE=epoch

START_EPOCH=${START_EPOCH:-10}
INCREMENT=${INCREMENT:-10}

END_EPOCH=$((START_EPOCH + INCREMENT))
T_MAX=${END_EPOCH}
loss=nrmse
attn=scse

mnt_dir=/mnt/nfs/home/mvsousa/Research_Project_Biomass_Prediction/1st-place-biomassters
data_dir=$mnt_dir/data
chkps_dir=$mnt_dir/models

# used backbones = [vgg16, resnet18, resnet50t, mobilenetv2_140, tf_efficientnetv2_l_in21k, vit_base_patch16_224_in21k]
backbone=${BACKBONE:-vgg16}
BS=1
FOLD=0

echo "--> Iniciando o job para o backbone: ${backbone}"
echo "--> Da época ${START_EPOCH} até ${END_EPOCH}"

CHECKPOINT_LOAD=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${START_EPOCH}"_"${loss}"_devscse_attnlin_augs_decplus7
CHECKPOINT=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${END_EPOCH}"_"${loss}"_devscse_attnlin_augs_decplus7

MASTER_PORT="${PORT}" CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node="${N_GPUS}" \
    $mnt_dir/src/train.py \
        --train-df $data_dir/features_metadata.csv \
        --train-images-dir $data_dir/train_features \
        --train-labels-dir $data_dir/train_agbm \
        --backbone "${backbone}" \
        --loss "${loss}" \
        --in-channels 15 \
        --optim "${OPTIM}" \
        --learning-rate "${LR}" \
        --weight-decay "${WD}" \
        --scheduler "${SCHEDULER}" \
        --T-max "${T_MAX}" \
        --num-epochs "${END_EPOCH}" \
        --checkpoint-dir "${CHECKPOINT}" \
        --fold "${FOLD}" \
        --scheduler-mode "${MODE}" \
        --batch-size "${BS}" \
        --load $CHECKPOINT_LOAD/model_last.pth \
        --augs \
        --dec-attn-type $attn \
        --dec-channels 384 368 352 336 320 \
        # --fp16 \



# LR=0.0001
# N_EPOCHS=0
# T_MAX=0
# CHECKPOINT_LOAD=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7
# CHECKPOINT=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7_plus800eb

# MASTER_PORT="${PORT}" CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node="${N_GPUS}" \
#     /mnt/nfs/home/mvsousa/the-biomassters/1st-place/src/train.py \
#         --train-df $data_dir/features_metadata.csv \
#         --train-images-dir $data_dir/train_features \
#         --train-labels-dir $data_dir/train_agbm \
#         --backbone "${backbone}" \
#         --loss "${loss}" \
#         --in-channels 15 \
#         --optim "${OPTIM}" \
#         --learning-rate "${LR}" \
#         --weight-decay "${WD}" \
#         --scheduler "${SCHEDULER}" \
#         --T-max "${T_MAX}" \
#         --num-epochs "${N_EPOCHS}" \
#         --checkpoint-dir "${CHECKPOINT}" \
#         --fold "${FOLD}" \
#         --scheduler-mode "${MODE}" \
#         --batch-size "${BS}" \
#         --load $CHECKPOINT_LOAD/model_last.pth \
#         --augs \
#         --dec-attn-type $attn \
#         --dec-channels 384 368 352 336 320 \
        # --fp16 \


# N_EPOCHS=0
# T_MAX=0
# CHECKPOINT_LOAD=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7_plus800eb
# CHECKPOINT=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7_plus800eb_100ft
# MASTER_PORT="${PORT}" CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node="${N_GPUS}" \
#     /mnt/nfs/home/mvsousa/the-biomassters/1st-place/src/train.py \
#         --train-df $data_dir/features_metadata.csv \
#         --train-images-dir $data_dir/train_features \
#         --train-labels-dir $data_dir/train_agbm \
#         --backbone "${backbone}" \
#         --loss "${loss}" \
#         --in-channels 15 \
#         --optim "${OPTIM}" \
#         --learning-rate "${LR}" \
#         --weight-decay "${WD}" \
#         --scheduler "${SCHEDULER}" \
#         --T-max "${T_MAX}" \
#         --num-epochs "${N_EPOCHS}" \
#         --checkpoint-dir "${CHECKPOINT}" \
#         --fold "${FOLD}" \
#         --scheduler-mode "${MODE}" \
#         --batch-size "${BS}" \
#         --load $CHECKPOINT_LOAD/model_last.pth \
#         --augs \
#         --dec-attn-type $attn \
#         --dec-channels 384 368 352 336 320 \
#         # --fp16 \
#         --ft \


# CHECKPOINT_LOAD=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7_plus800eb_100ft
# CHECKPOINT=$chkps_dir/"${backbone}"_f"${FOLD}"_b"${BS}"x"${N_GPUS}"_e"${N_EPOCHS}"_"${loss}"_devscse_attnlin_augs_decplus7_plus800eb_200ft
# MASTER_PORT="${PORT}" CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node="${N_GPUS}" \
#     /mnt/nfs/home/mvsousa/the-biomassters/1st-place/src/train.py \
#         --train-df $data_dir/features_metadata.csv \
#         --train-images-dir $data_dir/train_features \
#         --train-labels-dir $data_dir/train_agbm \
#         --backbone "${backbone}" \
#         --loss "${loss}" \
#         --in-channels 15 \
#         --optim "${OPTIM}" \
#         --learning-rate "${LR}" \
#         --weight-decay "${WD}" \
#         --scheduler "${SCHEDULER}" \
#         --T-max "${T_MAX}" \
#         --num-epochs "${N_EPOCHS}" \
#         --checkpoint-dir "${CHECKPOINT}" \
#         --fold "${FOLD}" \
#         --scheduler-mode "${MODE}" \
#         --batch-size "${BS}" \
#         --load $CHECKPOINT_LOAD/model_last.pth \
#         --augs \
#         --dec-attn-type $attn \
#         --dec-channels 384 368 352 336 320 \
#         # --fp16 \
#         --ft \
