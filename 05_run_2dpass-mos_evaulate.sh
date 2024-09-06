#!/bin/bash

prev_config="prev_config/05_run_2dpass-mos_evaulate.yaml"
is_prev=false

if [ -f "$prev_config" ]; then
    prev_data_path=$(grep "data_path:" $prev_config | awk '{print $2}')
    prev_pred_path=$(grep "pred_path:" $prev_config | awk '{print $2}')
    prev_ds_type=$(grep "ds_type:" $prev_config | awk '{print $2}')
    
    echo "Previous Configuration:"
    echo " - Dataset Path: $prev_data_path"
    echo " - Prediction Path: $prev_pred_path"
    echo " - Dataset Type: $prev_ds_type"
    
    read -p "Load previous config? [y/N]: " is_prev_input
    if [[ "$is_prev_input" =~ ^[Yy]$ ]]; then
        is_prev=true
    fi
fi

if [ "$is_prev" = false ]; then
    read -e -p "Enter the dataset path: " data_path
    read -e -p "Enter the prediction path: " pred_path
    read -p "Enter dataset type [valid, test]: " ds_type
    
    echo "data_path: $data_path" > $prev_data_path
    echo "pred_path: $pred_path" >> $prev_pred_path
    echo "ds_type: $ds_type" >> $prev_ds_type
else
    data_path=$prev_data_path
    pred_path=$prev_pred_path
    ds_type=$prev_ds_type
fi

python utils/evaluate_mos.py \
	-d $data_path \
	-p $pred_path \
	-s $ds_type

