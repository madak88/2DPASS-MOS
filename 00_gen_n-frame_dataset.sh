#!/bin/bash

prev_config="prev_config/00_gen_n-frame_dataset.yaml"
is_prev=false

if [ -f "$prev_config" ]; then
    prev_in_path=$(grep "in_path:" $prev_config | awk '{print $2}')
    prev_out_path=$(grep "out_path:" $prev_config | awk '{print $2}')
    prev_merge_size=$(grep "merge_size:" $prev_config | awk '{print $2}')
    prev_ds_type=$(grep "ds_type:" $prev_config | awk '{print $2}')
    
    echo "Previous Configuration:"
    echo " - Input Path: $prev_in_path"
    echo " - Output Path: $prev_out_path"
    echo " - Merge Size: $prev_merge_size"
    echo " - Dataset Type: $prev_ds_type"
    
    read -p "Load previous config? [y/N]: " is_prev_input
    if [[ "$is_prev_input" =~ ^[Yy]$ ]]; then
        is_prev=true
    fi
fi

if [ "$is_prev" = false ]; then
    read -e -p "Enter the original dataset path: " in_path
    read -e -p "Enter the output dataset path: " out_path
    read -p "Enter merge size: " merge_size
    read -p "Enter dataset type [train, val, test, all]: " ds_type
    
    echo "in_path: $in_path" > $prev_config
    echo "out_path: $out_path" >> $prev_config
    echo "merge_size: $merge_size" >> $prev_config
    echo "ds_type: $ds_type" >> $prev_config
else
    in_path=$prev_in_path
    out_path=$prev_out_path
    merge_size=$prev_merge_size
    ds_type=$prev_ds_type
fi

python gen_n-frame_dataset.py \
    --in_dataset "$in_path" \
    --out_dataset "$out_path" \
    --merge "$merge_size" \
    --split "$ds_type"

