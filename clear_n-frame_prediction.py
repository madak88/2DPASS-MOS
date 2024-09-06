
import os
import numpy as np
import argparse

def load_indexes(file_path):
    with open(file_path, 'r') as file:
        line = file.readline().strip()
        idx1, idx2 = map(int, line.split())
    return idx1, idx2

def clear_pred(pred_paths, indices_paths, output_path):
    all_pred_num = len(pred_paths)
    os.makedirs(output_path, exist_ok=True)
    for i in range(all_pred_num):
        act_pred = np.fromfile(pred_paths[i], dtype=np.uint32).reshape((-1, 1))
        start_idx, end_idx = load_indexes(indices_paths[i])
        act_pred[start_idx:end_idx].tofile(os.path.join(output_path, str(i).zfill(6)+".label"))
        print(f" - Clear merged predictions - {i}/{all_pred_num}", end='\r')

def get_file_paths(directory):
    paths = []
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            paths.append(os.path.join(dirpath, f))
    return paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clear specified ranges in prediction files and save the results.")
    parser.add_argument('--in_predictions', '-in1', type=str, help='Input Predictions Directory')
    parser.add_argument('--in_indeces', '-in2', type=str, help='Input Indeces Directory')
    parser.add_argument('--out_predictions', '-out', type=str, help='Output Predictions Directory')
    parser.add_argument('--split', '-s', type=str, default='val', help='DS type: train/val/test')
    args = parser.parse_args()

    ds_type = args.split
    assert ds_type in ['sk-train', 'sk-val', 'sk-test', 'sk-all', 'ap-val', 'ap-test', 'ap-all']
    if ds_type == "sk-train":  split = [0,1,2,3,4,5,6,7,9,10]
    if ds_type == "sk-val":    split = [8]
    if ds_type == "sk-test":   split = [11,12,13,14,15,16,17,18,19,20,21]
    if ds_type == "sk-all":    split = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    if ds_type == "ap-val":    split = [23,24,25]
    if ds_type == "ap-test":   split = [22,26]
    if ds_type == "ap-all":    split = [22,23,24,25,26]

    for seq_i in split:
        print("SEQUENCE", seq_i)
        # Get paths
        seq_str = "{0:02d}".format(int(seq_i))
        seq_path = os.path.join("sequences", seq_str)    
        pred_paths = get_file_paths(os.path.join(args.in_predictions, seq_path, "predictions"))
        indices_paths = get_file_paths(os.path.join(args.in_indeces, seq_path, "indices"))
        output_path = os.path.join(args.out_predictions, seq_path, "predictions")
        # Clear predictions
        clear_pred(pred_paths, indices_paths, output_path)
        print(f" - Clear merged predictions - {len(pred_paths)} done")

