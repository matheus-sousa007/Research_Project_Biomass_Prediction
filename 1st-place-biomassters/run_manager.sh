#!/bin/bash

# Lista de modelos
BACKBONES=("vgg16" "resnet18" "resnet50t" "mobilenetv2_140" "tf_efficientnetv2_l_in21k" "vit_base_patch16_224_in21k")

# Configuração das Parcelas
START_EPOCH=10
FINAL_TARGET=30
INCREMENT=10

# --- PREPARAÇÃO DO DICIONÁRIO DE IDs ---
# Isso cria um "mapa" para guardar o último ID de cada backbone separadamente
# Ex: LAST_IDS["vgg16"] = 1001, LAST_IDS["resnet50"] = 1002
declare -A LAST_IDS

# Inicializa o mapa vazio
for bb in "${BACKBONES[@]}"; do
    LAST_IDS[$bb]=""
done

# --- LOOP PRINCIPAL INVERTIDO ---
# 1. Loop Externo: Épocas (Avança o tempo)
for (( curr_start=$START_EPOCH; curr_start<$FINAL_TARGET; curr_start+=$INCREMENT )); do
    
    curr_end=$((curr_start + INCREMENT))
    
    echo "=== Preparando submissões para intervalo: $curr_start até $curr_end ==="

    # 2. Loop Interno: Backbones (Avança a largura/variedade)
    for bb in "${BACKBONES[@]}"; do
        
        JOB_NAME="${bb}_${curr_start}_to_${curr_end}"
        
        # Recupera o ID do job anterior ESPECÍFICO deste backbone
        PREV_ID=${LAST_IDS[$bb]}
        
        # Define a dependência
        if [ -z "$PREV_ID" ]; then
            DEP_FLAG=""
        else
            DEP_FLAG="-W depend=afterok:$PREV_ID"
        fi
        
        # Submete o job
        OUTPUT=$(qsub -N $JOB_NAME \
                      -v BACKBONE=$bb,START_EPOCH=$curr_start,INCREMENT=$INCREMENT \
                      $DEP_FLAG \
                      run.sh)
        
        # Limpa o output para pegar só o ID
        NEW_ID=$(echo $OUTPUT | grep -o '[0-9]*\.[a-z]*')
        if [ -z "$NEW_ID" ]; then NEW_ID=$OUTPUT; fi
        
        # SALVA O NOVO ID NO DICIONÁRIO PARA O PRÓXIMO LOOP DE ÉPOCAS
        LAST_IDS[$bb]=$NEW_ID
        
        echo "   -> Submetido: $bb (ID: $NEW_ID) [Dep: $PREV_ID]"
        
    done
    echo "---------------------------------------------------------------"
done