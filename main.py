#!/usr/bin/env python
#
# Sample code to inspect the data in 3D, fix the NOCS backgrounds and convert poses.
#
################################################################################## 
# Author: 
#   - Xavier Weber
#   - Email: corsmal-challenge@qmul.ac.uk
#
#  Created Date: 2022/11/23
#
# MIT License

# Copyright (c) 2022 CORSMAL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#--------------------------------------------------------------------------------

import utils
import argparse
import numpy as np
import open3d as o3d
import cv2 
import os
import json

def load_images(batch_folder, image_index):
    # depth
    depth_path = os.path.join(args.choc_dir, "mixed-reality", "depth", batch_folder, "{}.png".format(image_index)); print(depth_path)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:,:,2]
    # rgb
    rgb_path = os.path.join(args.choc_dir, "mixed-reality", "rgb", batch_folder, "{}.png".format(image_index))
    rgb = cv2.imread(rgb_path)[:,:,::-1] # bgr to rgb 
    # nocs
    nocs_path = os.path.join(args.choc_dir, "mixed-reality", "nocs", batch_folder, "{}.png".format(image_index))
    nocs = cv2.imread(nocs_path)[:,:,:3]
    nocs = nocs[:,:,::-1] # bgr to rgb 
    # mask
    mask_path = os.path.join(args.choc_dir, "mixed-reality", "mask", batch_folder, "{}.png".format(image_index))
    mask = cv2.imread(mask_path)[:,:,2]
    return rgb, depth, nocs, mask

def load_pose_scaling_factor(batch_folder, image_index):
    # get json about this image
    image_info_path = os.path.join(args.choc_dir, "mixed-reality", "annotations", batch_folder, "{}.json".format(image_index))
    with open(image_info_path, 'r') as f:
        image_info = json.load(f)
    
    # get json about objects
    f = open(os.path.join(args.choc_dir, "mixed-reality", "extra", "object_models", "object_datastructure.json"))
    objects_info = json.load(f)
    
    # get object id and dimensions
    object_id = image_info["object_id"]
    metric_width = objects_info["objects"][object_id]["width"]   # meters
    metric_height = objects_info["objects"][object_id]["height"] # meters
    metric_depth = objects_info["objects"][object_id]["depth"]   # meters
    
    # get pose from annotation file
    pose_quat_wxyz = image_info["pose_quaternion_wxyz"]
    location_xyz = image_info["location_xyz"]
    # convert it
    RT = utils.convert_blender_pose_to_camera_object_pose(pose_quat_wxyz, location_xyz, metric_height, verbose=True)
    
    # get scaling factor
    gt_scale_factor = utils.get_space_dag(metric_width, metric_height, metric_depth, scale=1000, verbose=True) # normalising factor (NOCS)

    return RT, gt_scale_factor

def visualise_in_3D(pts, metric_points, RT):

    # Depth - BLUE
    depth_pcl = o3d.geometry.PointCloud()
    depth_pcl.points = o3d.utility.Vector3dVector(pts)
    depth_pcl.paint_uniform_color([0, 0, 1]) # RGB 

    # Annotated pose - RED
    annotated_pcl = o3d.geometry.PointCloud()
    annotated_pts_3D = utils.transform_coordinates_3d(metric_points.transpose(), RT)
    annotated_pcl.points = o3d.utility.Vector3dVector(annotated_pts_3D.transpose())
    annotated_pcl.paint_uniform_color([1, 0, 0]) 

    # Metric pointcloud - GREEN
    metric_pcl = o3d.geometry.PointCloud()
    metric_pcl.points = o3d.utility.Vector3dVector(metric_points)
    #metric_pcl.colors = o3d.utility.Vector3dVector(GT_NOCS_points_og)
    metric_pcl.paint_uniform_color([0, 1, 0])

    # NOCS pointcloud - NOCS colors
    #nocs_pcl = o3d.geometry.PointCloud()
    #nocs_pcl.points = o3d.utility.Vector3dVector(nocs_points)
    #nocs_pcl.colors = o3d.utility.Vector3dVector(nocs_points)

    # Draw
    origin_axes_big = o3d.geometry.TriangleMesh.create_coordinate_frame(size=100, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([depth_pcl, metric_pcl, annotated_pcl, origin_axes_big], window_name="Annotated (R), Metric Object (G), Depth (B)")

if __name__ == '__main__':
    
    # Parsing arguments from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--choc_dir', type=str, help="Path to the CHOC dataset.", default="sample/CHOC")
    parser.add_argument('--image_index', type=str, help="Image index in 6-digit string format.", default="000251")
    parser.add_argument('--operation', type=str, help="inspect, fix_nocs, convert_pose", default="inspect")
    args = parser.parse_args()

    # Get the batch folder
    image_index = args.image_index
    batch_folder = utils.image_index_to_batch_folder(args.image_index)

    # Get the image files
    rgb, depth, nocs, mask = load_images(batch_folder, image_index)

    if args.operation == "inspect":
        # clean nocs background
        nocs_fixed = utils.fix_background_nocs(nocs)
        # binarize mask; remove hand too
        mask_binary = utils.mask_remove_hand_and_binarize(mask)
        # remove pixels from nocs and mask if they are not both non-zero (both meaning: both in nocs and mask)
        nocs_clean, mask_clean, gt_image_points = utils.intersect_nocs_mask_values(nocs_fixed, mask_binary)
        # normalise nocs
        nocs_norm = np.array(nocs_clean, dtype=np.float32) / 255.0
        # Back-project depth points | image plane (u,v,Z) -> camera coordinate system (X,Y,Z))
        pts, idxs = utils.backproject_opengl(depth, utils.get_intrinsics(), mask_clean)

        # Get pose and scaling factor
        RT, gt_scale_factor = load_pose_scaling_factor(batch_folder, image_index)

        # Get the nocs points
        nocs_points = nocs_norm[idxs[0], idxs[1], :] - 0.5
        # Un-normalise to get the metric points
        metric_points = nocs_points * gt_scale_factor # in millimeter

        visualise_in_3D(pts, metric_points, RT)
    
   
    elif args.operation == "fix_nocs":
        
        # Visualise the problem [zoom in on background to see the pixel values]
        cv2.imshow("problematic nocs background", nocs[:,:,::-1])

        # Fix the problem
        print("unique values in nocs:", np.unique(nocs))
        nocs_fixed = utils.fix_background_nocs(nocs)
        print("unique values in nocs fixed:", np.unique(nocs_fixed))

        # Visualise the fixed nocs [zoom in on background to see the pixel values]
        cv2.imshow("fixed nocs background", nocs_fixed[:,:,::-1])

        # Press 'Esc' key to close windows
        while True:
            k = cv2.waitKey(0) & 0xFF
            print(k)
            if k == 27:
                cv2.destroyAllWindows()
                break


    elif args.operation == "convert_pose":
        
        RT, _ = load_pose_scaling_factor(batch_folder, image_index)
    
    else:
        raise Exception("Unknown operation:", args.operation)