#!/bin/bash

prev_config="prev_config/04_run_add-semantic.yaml"
is_prev=false

if [ -f "$prev_config" ]; then
    prev_ds_type=$(grep "ds_type:" $prev_config | awk '{print $2}')
    prev_data_path=$(grep "data_path:" $prev_config | awk '{print $2}')
    prev_mos_pred_path=$(grep "mos_pred_path:" $prev_config | awk '{print $2}')
    prev_sem_pred_path=$(grep "sem_pred_path:" $prev_config | awk '{print $2}')
    prev_out_pred_path=$(grep "out_pred_path:" $prev_config | awk '{print $2}')
    prev_frame_num=$(grep "frame_num:" $prev_config | awk '{print $2}')
    
    echo "Previous Configuration:"
    echo " - Dataset Type: $prev_ds_type"
    echo " - Dataset Path: $prev_data_path"
    echo " - MOS prediction path: $prev_mos_pred_path"
    echo " - Semantic prediction path: $prev_sem_pred_path"
    echo " - Output prediction path: $prev_out_pred_path"
    echo " - Frame Number: $prev_frame_num"
    
    read -p "Load previous config? [y/N]: " is_prev_input
    if [[ "$is_prev_input" =~ ^[Yy]$ ]]; then
        is_prev=true
    fi
fi

if [ "$is_prev" = false ]; then
    read -p "Enter dataset type [val, test]: " ds_type
    read -e -p "Enter the dataset path: " data_path
    read -e -p "Enter the MOS prediction path: " mos_pred_path
    read -e -p "Enter the Semantic prediction path: " sem_pred_path
    read -e -p "Enter the Output prediction path: " out_pred_path
    read -p "Enter the Frame Number: " frame_num
    
    echo "ds_type: $ds_type" > $prev_ds_type
    echo "data_path: $data_path" >> $prev_data_path
    echo "mos_pred_path: $mos_pred_path" >> $prev_mos_pred_path
    echo "sem_pred_path: $sem_pred_path" >> $prev_sem_pred_path
    echo "out_pred_path: $out_pred_path" >> $prev_out_pred_path
    echo "frame_num: $frame_num" >> $prev_frame_num
else
    ds_type=$prev_ds_type
    data_path=$prev_data_path
    mos_pred_path=$prev_mos_pred_path
    sem_pred_path=$prev_sem_pred_path
    out_pred_path=$prev_out_pred_path
    frame_num=$prev_frame_num
fi

python add_semantic.py \
	-s $ds_type \
	-id $data_path \
	-im $mos_pred_path \
	-is $sem_pred_path \
	-om $out_pred_path \
	-fn $frame_num 

