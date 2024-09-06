import os
import numpy as np
import torch
import math
import shutil
import argparse

# Load ground truth poses from file.
def load_poses(pose_path):
    all_poses = []
    try:
        with open(pose_path, "r") as f: lines = f.readlines()
        for line in lines:
            act_poses = np.fromstring(line, dtype=float, sep=" ")
            if len(act_poses) == 12:
                act_poses = act_poses.reshape(3, 4)
                act_poses = np.vstack((act_poses, [0, 0, 0, 1]))
            elif len(act_poses) == 16:
                act_poses = act_poses.reshape(4, 4)
            all_poses.append(act_poses)
    except FileNotFoundError:
        print("Ground truth poses are not avaialble.")
    return np.array(all_poses)

# Load calib from file.
def load_calib(calib_path):
    T_cam_velo = []
    try:
        with open(calib_path, "r") as f: lines = f.readlines()
        for line in lines:
            if "Tr:" in line:
                line = line.replace("Tr:", "")
                T_cam_velo = np.fromstring(line, dtype=float, sep=" ")
                T_cam_velo = T_cam_velo.reshape(3, 4)
                T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))
    except FileNotFoundError:
        print("Calibrations are not avaialble.")
    return np.array(T_cam_velo)

# Create the poses by the given path
def get_poses(path_to_seq):
    # Load Poses
    pose_file = os.path.join(path_to_seq, "poses.txt")
    calib_file = os.path.join(path_to_seq, "calib.txt")
    poses = load_poses(pose_file)
    inv_frame0 = np.linalg.inv(poses[0])
    # Load calib
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = T_cam_velo.reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)
    # Convert from camera coord to LiDAR coord
    new_poses = []
    for pose in poses: new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
    poses = np.array(new_poses)
    return poses

def transform_point_cloud(past_point_clouds, from_pose, to_pose):
    transformation = torch.Tensor(np.linalg.inv(to_pose) @ from_pose)
    NP = past_point_clouds.shape[0]
    xyz1 = torch.hstack((past_point_clouds, torch.ones(NP, 1))).T
    past_point_clouds = (transformation @ xyz1).T[:, :3]
    return past_point_clouds

def save_indices(file_path, recovery_indices):
    with open(file_path, 'w') as file:
        file.write(f"{recovery_indices[0]} {recovery_indices[1]}")

def merge(merge_size, original_idx, velo_paths, labels_paths, poses, output_path):
    seq_size = len(velo_paths)
    there_are_labels = len(labels_paths) != 0
    all_transformed_points = None
    if there_are_labels: all_gt_labels = None
    for act_idx in range(merge_size):
        act_idx = original_idx - math.floor(merge_size/2) + act_idx
        if act_idx < 0 or act_idx >= seq_size: continue
        raw_velo_data = np.fromfile(velo_paths[act_idx], dtype=np.float32).reshape((-1, 4))
        tensor_raw_velo_data = torch.tensor(raw_velo_data)
        tensor_points = tensor_raw_velo_data[:,:3]
        remissions = tensor_raw_velo_data[:,3]
        remissions = remissions.unsqueeze(1)
        from_pose = poses[act_idx]
        to_pose = poses[original_idx]
        transformed_tensor_points = transform_point_cloud(tensor_points, from_pose, to_pose)
        transformed_tensor_points = torch.cat((transformed_tensor_points,remissions),1)
        transformed_points = np.array(transformed_tensor_points)
        if all_transformed_points is None: 
            all_transformed_points = transformed_points
            if act_idx == original_idx: recovery_indices = [0, len(transformed_points)]
        else: 
            if act_idx == original_idx: recovery_indices = [len(all_transformed_points)]
            all_transformed_points = np.append(all_transformed_points, transformed_points, axis=0)
            if act_idx == original_idx: recovery_indices.append(len(all_transformed_points))
        if there_are_labels:
            gt_labels= np.fromfile(labels_paths[act_idx], dtype=np.uint32).reshape((-1, 1))
            if all_gt_labels is None: all_gt_labels = gt_labels
            else: all_gt_labels = np.append(all_gt_labels, gt_labels, axis=0)
    velo_path = os.path.join(output_path,"velodyne")
    if not os.path.exists(velo_path):os.makedirs(velo_path)
    all_transformed_points.tofile(os.path.join(velo_path, str(original_idx).zfill(6)+".bin"))
    if there_are_labels:
        labels_path = os.path.join(output_path,"labels")
        if not os.path.exists(labels_path):os.makedirs(labels_path)
        all_gt_labels.tofile(os.path.join(labels_path, str(original_idx).zfill(6)+".label"))
    indices_path = os.path.join(output_path,"indices")
    if not os.path.exists(indices_path):os.makedirs(indices_path)
    index_path = os.path.join(indices_path, str(original_idx).zfill(6)+".index")
    save_indices(index_path, recovery_indices)

def file_paths(directory):
    paths = []
    if not os.path.exists(directory): return paths
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            paths.append(os.path.join(dirpath, f))
    return paths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LiDAR point clouds and labels.")
    parser.add_argument('--in_dataset', '-in', type=str, help='Input Dataset Directory')
    parser.add_argument('--out_dataset', '-out', type=str, help='Output Dataset Directory')
    parser.add_argument('--merge', '-m', type=int, default=2, help='Merge Size')
    parser.add_argument('--split', '-s', type=str, default='val', help='DS type: train/val/test/all')
    args = parser.parse_args()

    merge_size = args.merge

    ds_type = args.split
    assert ds_type in ['train', 'val', 'test', 'all']
    if ds_type == "train":  split = [0,1,2,3,4,5,6,7,9,10]
    if ds_type == "val":    split = [8]
    if ds_type == "test":   split = [11,12,13,14,15,16,17,18,19,20,21]
    if ds_type == "all":    split = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

    for seq_i in split:
        print("SEQUENCE", seq_i)
        # Get paths
        seq_str = "{0:02d}".format(int(seq_i))
        seq_path = os.path.join("sequences", seq_str)
        input_path = os.path.join(args.in_dataset, seq_path)
        output_path = os.path.join(args.out_dataset, seq_path)
        if not os.path.exists(output_path): os.makedirs(output_path)
        # Copy pose, calib and image files
        shutil.copy(os.path.join(input_path, "poses.txt"), output_path)
        print(" - Copy poses - done")
        shutil.copy(os.path.join(input_path, "calib.txt"), output_path)
        print(" - Copy calib - done")
        shutil.copytree(os.path.join(input_path, "image_2"), os.path.join(output_path, "image_2"))
        print(" - Copy image_2 - done")
        # Get paths
        velo_paths = file_paths(os.path.join(input_path, "velodyne"))
        labels_paths = file_paths(os.path.join(input_path, "labels"))
        poses = get_poses(os.path.join(input_path))
        # Create merge dataset
        for i in range(len(poses)):
            merge(merge_size, i, velo_paths, labels_paths, poses, output_path)
            print(f" - Generate premerge frames - {i}/{len(poses)}", end='\r')
        print(f" - Generate premerge frames - {len(poses)} done")

