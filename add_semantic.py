
import os
import argparse
import numpy as np
import open3d as o3d

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
    transformation = np.linalg.inv(to_pose) @ from_pose
    NP = past_point_clouds.shape[0]
    xyz1 = np.hstack((past_point_clouds, np.ones((NP, 1))))
    transformed_points = (transformation @ xyz1.T).T[:, :3]
    return transformed_points

def file_paths(directory):
    paths = []
    if not os.path.exists(directory): return paths
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            paths.append(os.path.join(dirpath, f))
    return paths

def cluster(pred_lab_up, velo_data, mov_idx, sem_idx, min_distance, min_points, dyn_thresh_percent, thresh_point):
    if len(sem_idx) == 0: return pred_lab_up
    # Select the points of required class and cluster them based on distance
    pt_cloud = o3d.geometry.PointCloud()
    pt_cloud.points = o3d.utility.Vector3dVector(velo_data[sem_idx])

    labels = np.array(pt_cloud.cluster_dbscan(eps=min_distance, min_points=min_points))
    num_clusters = labels.max() + 1

    # Select each cluster and fit a cuboid to each cluster
    for num in range(num_clusters):
        label_idx = np.where(labels == num)[0]
        # Ignore cluster that has points less than threshpoint points.
        if len(label_idx) < thresh_point: continue
        # Count how many percent of the given instance has moving label 
        is_dyn = len(set(mov_idx) & set(sem_idx[label_idx])) / len(sem_idx[label_idx])
        if is_dyn > dyn_thresh_percent:
            if type(pred_lab_up) == np.ndarray: pred_lab_up = pred_lab_up.tolist()
            pred_lab_up.extend(sem_idx[label_idx])

    # Write the new labels into files
    return np.unique(np.hstack([mov_idx, pred_lab_up]))

def main(args):

    frame_num = args.frame_num

    ds_type = args.split

    assert ds_type in ['sk-train', 'sk-val', 'sk-test', 'sk-all']
    if ds_type == "sk-train":  split = [0,1,2,3,4,5,6,7,9,10]
    if ds_type == "sk-val":    split = [8]
    if ds_type == "sk-test":   split = [11,12,13,14,15,16,17,18,19,20,21]
    if ds_type == "sk-all":    split = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

    for seq_i in split:
        print("SEQUENCE", seq_i)

        seq_str = "{0:02d}".format(int(seq_i))
        seq_path = os.path.join("sequences", seq_str)

        velo_paths = file_paths(os.path.join(args.in_dataset, seq_path, 'velodyne'))
        moving_paths = file_paths(os.path.join(args.in_moving, seq_path, 'predictions'))
        semantic_paths = file_paths(os.path.join(args.in_semantic, seq_path, 'predictions'))
        poses = get_poses(os.path.join(args.in_dataset, seq_path))

        output_dir = os.path.join(args.out_moving, seq_path, 'predictions')
        if not os.path.exists(output_dir):os.makedirs(output_dir)

        min_distance = args.min_dist
        min_points = args.min_points
        dyn_thresh_percent = args.dyn_thresh_percent
        thresh_point = args.tresh_point

        seq_size = len(poses)

        for ori_frame_idx in range(seq_size):

            all_velo_data = np.array([])
            all_moving_data = np.array([])
            all_semantic_data = np.array([])

            for act_frame_idx in range(frame_num):
                act_frame_idx = ori_frame_idx-act_frame_idx
                if act_frame_idx < 0 : continue
                # Read velodyne files and store them
                raw_velo_data = np.fromfile(velo_paths[act_frame_idx], dtype=np.float32).reshape((-1, 4))
                points = raw_velo_data[:,:3]
                from_pose = poses[act_frame_idx]
                to_pose = poses[ori_frame_idx]
                veo_data = transform_point_cloud(points, from_pose, to_pose)
                all_velo_data = np.vstack([all_velo_data, veo_data]) if all_velo_data.size else veo_data
                # Read predicted moving object labels and find them
                moving_data = np.fromfile(moving_paths[act_frame_idx], dtype=np.uint32).reshape((-1, 1))
                all_moving_data = np.vstack([all_moving_data, moving_data]) if all_moving_data.size else moving_data
                # Read predicted semantic labels and find them
                semantic_data = np.fromfile(semantic_paths[act_frame_idx], dtype=np.uint32).reshape((-1, 1))
                all_semantic_data = np.vstack([all_semantic_data, semantic_data]) if all_semantic_data.size else semantic_data
                
            moving_idx = np.where(all_moving_data == 251)[0]

            car_idx = np.where(all_semantic_data == 10)[0]
            #cycle_idx = np.where(all_semantic_data == 11)[0]
            #cyclist_idx = np.where(all_semantic_data == 31)[0]
            #mcycle_idx = np.where(all_semantic_data == 15)[0]
            #mcyclist_idx = np.where(all_semantic_data == 32)[0]
            #bus_idx = np.where(all_semantic_data == 13)[0]
            #rails_idx = np.where(all_semantic_data == 16)[0]
            #truck_idx = np.where(all_semantic_data == 18)[0]
            #oth_idx = np.where(all_semantic_data == 20)[0]

            pred_lab_up = []
            pred_lab_up = cluster(pred_lab_up, all_velo_data, moving_idx, car_idx, min_distance, min_points, dyn_thresh_percent, thresh_point)
            #pred_lab_up = cluster(pred_lab_up, all_velo_data, moving_idx, cycle_idx, min_distance, min_points, dyn_thresh_percent, thresh_point)
            #pred_lab_up = cluster(pred_lab_up, all_velo_data, moving_idx, cyclist_idx, min_distance, min_points, dyn_thresh_percent, thresh_point)
            #pred_lab_up = cluster(pred_lab_up, all_velo_data, moving_idx, mcycle_idx, min_distance, min_points, dyn_thresh_percent, thresh_point)
            #pred_lab_up = cluster(pred_lab_up, all_velo_data, moving_idx, mcyclist_idx, min_distance, min_points, dyn_thresh_percent, thresh_point)
            #pred_lab_up = cluster(pred_lab_up, all_velo_data, moving_idx, bus_idx, min_distance, min_points, dyn_thresh_percent, thresh_point)
            #pred_lab_up = cluster(pred_lab_up, all_velo_data, moving_idx, rails_idx, min_distance, min_points, dyn_thresh_percent, thresh_point)
            #pred_lab_up = cluster(pred_lab_up, all_velo_data, moving_idx, truck_idx, min_distance, min_points, dyn_thresh_percent, thresh_point)
            #pred_lab_up = cluster(pred_lab_up, all_velo_data, moving_idx, oth_idx, min_distance, min_points, dyn_thresh_percent, thresh_point)

            raw_velo_data = np.fromfile(velo_paths[ori_frame_idx], dtype=np.float32).reshape((-1, 4))
            xyz = raw_velo_data[:,:3]
            pred_lab_up = pred_lab_up[pred_lab_up < xyz.shape[0]]

            moving_data = np.fromfile(moving_paths[ori_frame_idx], dtype=np.uint32).reshape((-1, 1))

            # Write the new labels into files
            new_labels = np.full_like(moving_data, 9)
            new_labels[np.where(moving_data == 0)] = 0
            pred_lab_up = pred_lab_up.astype(int)
            new_labels[pred_lab_up] = 251

            file_num = "{0:06d}".format(int(ori_frame_idx))
            output_path = os.path.join(output_dir, file_num + ".label")

            new_labels.tofile(output_path)
            print(f" - Add semantic - {ori_frame_idx}/{seq_size}", end='\r')

        print(f" - Add semantic - {seq_size} done")


if __name__ == "__main__":

# = Default parameters ============================================================================

    def_split = 'sk-val'

    root = '/home'
    def_data_path = os.path.join(root, 'ubuntu', 'downloads', 'kitti-stuff', 'datasets', 'calib-color-labels-velodyne_all', 'dataset') #, 'sequences', ...)
    def_moving_path = os.path.join(root, 'ubuntu', 'downloads', 'kitti-stuff', 'predictions', 'pred_sk-val_2dpass-mos_frames-02_downsampling-00_tta-12') #, 'sequences', ...)
    def_semantic_path = os.path.join(root, 'ubuntu', 'downloads', 'kitti-stuff', 'predictions', 'pred_sk-val_2dpass_original_tta-12') #, 'sequences', ...)
    def_output_path = os.path.join(root, 'ubuntu', 'downloads', 'kitti-stuff', 'predictions', 'pred_sk-val_2dpass-mos_frames-02_tta-12_sem3') #, 'sequences', ...)

    def_min_dist = 0.5
    def_min_points = 10
    def_dyn_thresh_percent = 0.4
    def_thresh_point = 200

    def_frame_num = 2

# =================================================================================================

    parser = argparse.ArgumentParser(description="Add semantic to MOS")

    parser.add_argument('--split', '-s', type=str, default=def_split, help='DS type: sk-train/sk-val/sk-test/sk-all')

    parser.add_argument('--in_dataset', '-id', type=str, default=def_data_path, help='Path for the dataset directory')
    parser.add_argument('--in_moving', '-im', type=str, default=def_moving_path, help='Path for moving labels')
    parser.add_argument('--in_semantic', '-is', type=str, default=def_semantic_path, help='Path for labels of semantic segmentation')
    parser.add_argument('--out_moving', '-om', type=str, default=def_output_path, help='Output path for new labels')

    parser.add_argument('--min_dist', '-md', type=float, default=def_min_dist, help='Parameter of cluster extraction')
    parser.add_argument('--min_points', '-mp', type=int, default=def_min_points, help='Parameter of cluster extraction')
    parser.add_argument('--dyn_thresh_percent', '-dtp', type=float, default=def_dyn_thresh_percent, help='Percentage of minimum moving labeled points of an object to consider as moving object')
    parser.add_argument('--tresh_point', '-tp', type=int, default=def_thresh_point, help='Minimum points to consider point cluster as object')

    parser.add_argument('--frame_num', '-fn', type=int, default=def_frame_num, help='Number of frames to merge')

    args = parser.parse_args()

    main(args)
