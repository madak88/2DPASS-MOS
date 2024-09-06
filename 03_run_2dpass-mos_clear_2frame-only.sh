#!/bin/bash

prev_config="prev_config/03_run_2dpass-mos_clear_2frame-only.yaml"
is_prev=false

if [ -f "$prev_config" ]; then
    prev_pred_path=$(grep "pred_path:" $prev_config | awk '{print $2}')
    prev_inde_path=$(grep "inde_path:" $prev_config | awk '{print $2}')
    prev_out_path=$(grep "out_path:" $prev_config | awk '{print $2}')
    prev_ds_type=$(grep "ds_type:" $prev_config | awk '{print $2}')
    
    echo "Previous Configuration:"
    echo " - Prediction Path: $prev_pred_path"
    echo " - Index Path: $prev_inde_path"
    echo " - Output Path: $prev_out_path"
    echo " - Dataset Type: $prev_ds_type"
    
    read -p "Load previous config? [y/N]: " is_prev_input
    if [[ "$is_prev_input" =~ ^[Yy]$ ]]; then
        is_prev=true
    fi
fi

if [ "$is_prev" = false ]; then
    read -e -p "Enter predictions path: " pred_path
    read -e -p "Enter data path: " inde_path
    read -e -p "Enter save location path (and a folder name): " out_path # Example: output/1frame-12tta
    read -p "Enter dataset type [val, test]: " ds_type
    
    echo "pred_path: $pred_path" > $prev_config
    echo "inde_path: $inde_path" >> $prev_config
    echo "out_path: $out_path" >> $prev_config
    echo "ds_type: $ds_type" >> $prev_config
else
    pred_path=$prev_pred_path
    inde_path=$prev_inde_path
    out_path=$prev_out_path
    ds_type=$prev_ds_type
fi

python clear_n-frame_prediction.py \
    --in_predictions $pred_path \
    --in_indeces $inde_path \
    --out_predictions $out_path \
    --split $ds_type

rm -rf $pred_path

