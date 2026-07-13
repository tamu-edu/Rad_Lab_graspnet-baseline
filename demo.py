""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
from rad_lab_hct_zbridge import bridgeServer, bridge_pb2

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path, weights_only=False)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net

def get_and_process_data(data_dir):
    # load data
    color = np.array(Image.open(os.path.join(data_dir, 'color.png')), dtype=np.float32) / 255.0
    depth = np.array(Image.open(os.path.join(data_dir, 'depth.png')))
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    #meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    #intrinsic = meta['intrinsic_matrix']
    #factor_depth = meta['factor_depth']

    # generate cloud

    # HARDCODE YOUR ZED INTRINSICS DIRECTLY:
    fx = 732.146484375
    fy = 732.146484375
    cx = 638.7239379882812
    cy = 360.1290588378906
    factor_depth = 1000.0  # Tells GraspNet that 1000 pixels = 1 meter

    #camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    camera = CameraInfo(1280.0, 720.0, fx,fy,cx,cy, factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

    # get valid points
    mask = (workspace_mask > 0) & (depth > 0)
    cloud_masked = cloud[mask]
    color_masked = color[mask]

    tmp_pcd = o3d.geometry.PointCloud()
    tmp_pcd.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float64))

    _, inlier_indices = tmp_pcd.remove_statistical_outlier(
        nb_neighbors=30, 
        std_ratio=1
    )
    
    # Filter arrays using generated inlier index mask
    cloud_masked = cloud_masked[inlier_indices]
    color_masked = color_masked[inlier_indices]

    pcd_for_ransac = o3d.geometry.PointCloud()
    pcd_for_ransac.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float64))
    
    plane_model, inliers = pcd_for_ransac.segment_plane(
        distance_threshold=0.010,  # 10mm table threshold
        ransac_n=3,
        num_iterations=200
    )
    
    # Generate a boolean mask where True means "NOT part of the table plane"
    #non_table_mask = np.ones(len(cloud_masked), dtype=bool)
    #non_table_mask[inliers] = False
    
    # Strip the table entirely out of the tracking arrays
    #cloud_masked = cloud_masked[non_table_mask]
    #color_masked = color_masked[non_table_mask]

    # 3. NEW: Quick Secondary Clean to remove tiny floating artifacts left near table-cut boundaries
    #pcd_clean = o3d.geometry.PointCloud()
    #pcd_clean.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float64))
    #_, post_inliers = pcd_clean.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.5)
    
    #cloud_masked = cloud_masked[post_inliers]
    #color_masked = color_masked[post_inliers]

    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:25]
    grippers = gg.to_open3d_geometry_list()

    for i in range(min(25, len(grippers))):
        grippers[i].paint_uniform_color([0.0, 0.0, 1.0]) 

    # Paint the remaining backup grasps RED (Danger / Lower Priority)
    for i in range(25, len(grippers)):
        grippers[i].paint_uniform_color([1.0, 0.0, 0.0])
    grippers[0].paint_uniform_color([0.0,1.0,0.0])
    o3d.visualization.draw_geometries([cloud, *grippers])


def demo(data_dir):
    #server = bridgeServer(port=5555)
    #server.start(handle_incoming_request)

    
    
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    pts = np.asarray(cloud.points)
    print("min:", pts.min(axis=0))
    print("max:", pts.max(axis=0))
    print("extent:", pts.max(axis=0) - pts.min(axis=0))
    gg = get_grasps(net, end_points)

    gg.nms()
    gg.sort_by_score()

    for i in range(10):
        print(
            i,
            gg[i].score,
            gg[i].translation
        )
    
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    vis_grasps(gg, cloud)

if __name__=='__main__':
    data_dir = '/home/orin/testing/zed_test/my_zed_data'
    demo(data_dir)
