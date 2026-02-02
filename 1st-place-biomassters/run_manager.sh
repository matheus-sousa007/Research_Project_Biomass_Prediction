#!/bin/bash

BACKBONES=("vgg16" "resnet18" "resnet50t" "mobilenetv2_140" "tf_efficientnetv2_l_in21k" "vit_base_patch16_224_in21k")

START_EPOCH=10
FINAL_TARGET=30
INCREMENT=10

declare -A LAST_IDS

for bb in "${BACKBONES[@]}"; do
    LAST_IDS[$bb]=""
done

for (( curr_start=$START_EPOCH; curr_start<$FINAL_TARGET; curr_start+=$INCREMENT )); do
    
    curr_end=$((curr_start + INCREMENT))
    
    echo "=== Preparando submissões para intervalo: $curr_start até $curr_end ==="

    for bb in "${BACKBONES[@]}"; do
        
        JOB_NAME="${bb}_${curr_start}_to_${curr_end}"
        
        PREV_ID=${LAST_IDS[$bb]}
        
        if [ -z "$PREV_ID" ]; then
            DEP_FLAG=""
        else
            DEP_FLAG="-W depend=afterok:$PREV_ID"
        fi
        
        OUTPUT=$(qsub -N $JOB_NAME \
                      -v BACKBONE=$bb,START_EPOCH=$curr_start,INCREMENT=$INCREMENT \
                      $DEP_FLAG \
                      run.sh)
        
        NEW_ID=$(echo $OUTPUT | grep -o '[0-9]*\.[a-z]*')
        if [ -z "$NEW_ID" ]; then NEW_ID=$OUTPUT; fi
        
        LAST_IDS[$bb]=$NEW_ID
        
        echo "   -> Submetido: $bb (ID: $NEW_ID) [Dep: $PREV_ID]"
        
    done
    echo "---------------------------------------------------------------"
done