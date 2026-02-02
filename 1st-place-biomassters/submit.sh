#!/usr/bin/env sh
#PBS -N 1st-place-biomassters-test-predictions
#PBS -l select=1:ncpus=1:ngpus=1:mem=1gb
#PBS -l walltime=00:30:00

conda init 
source /mnt/nfs/home/mvsousa/.bashrc
conda env create -f /mnt/nfs/home/mvsousa/the-biomassters/1st-place/environment.yml
conda activate .1st-place-biomassters

mnt_dir=/mnt/nfs/home/mvsousa/Research_Project_Biomass_Prediction/1st-place-biomassters
data_dir=$mnt_dir/data
chkps_dir=$mnt_dir/models
preds_dir=$mnt_dir/preds

python \
    $mnt_dir/src/submit.py \
    --test-df $data_dir/features_metadata.csv \
    --test-images-dir $data_dir/test_features \
    --model-path $chkps_dir/tf_efficientnetv2_l_in21k_f0_b8x2_e100_nrmse_devscse_attnlin_augs_decplus7_plus800eb_200ft/modelo_best.pth \
    --tta 4 \
    --batch-size 16 \
    --out-dir $preds_dir \
