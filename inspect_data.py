#!/usr/bin/env python
#
# Sample code to view the object, transformed object, depth pointclouds in Open3D, per image in CHOC mixed-reality.
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

# Parsing arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--choc_dir', type=str, help="Path to the CHOC dataset.")
parser.add_argument('--image_index', type=str, help="Image index in 6-digit string format.", default="000001")
args = parser.parse_args()

# Get the batch folder
image_index = args.image_index
batch_folder = utils.image_index_to_batch_folder(args.image_index)

### Load ground truths
# depth
depth_path = os.path.join(args.choc_dir, "mixed-reality", "depth", batch_folder, "{}.png".format(image_index)); print(depth_path)
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)[:,:,2]
# rgb
rgb_path = os.path.join(args.choc_dir, "mixed-reality", "rgb", batch_folder, "{}.png".format(image_index))
rgb = cv2.imread(rgb_path)[:,:,::-1]
# nocs
nocs_path = os.path.join(args.choc_dir, "mixed-reality", "nocs", batch_folder, "{}.png".format(image_index))
nocs = cv2.imread(nocs_path)[:,:,:3]
nocs = nocs[:,:,(2, 1, 0)] # bgr to rgb 
# mask
mask_path = os.path.join(args.choc_dir, "mixed-reality", "mask", batch_folder, "{}.png".format(image_index))
mask = cv2.imread(mask_path)[:,:,2]
gt_class_pixelID = np.unique(mask)[1]
gt_class_ID = utils.map_pixelValue_to_classId(gt_class_pixelID)

#### Process data
# clean nocs background
nocs_fixed = utils.fix_background_nocs(nocs)
# binarize mask; remove hand too
mask_binary = utils.mask_remove_hand_and_binarize(mask)
# remove pixels from nocs and mask if they are not both non-zero (both meaning: both in nocs and mask)
nocs_clean, mask_clean, gt_image_points = utils.intersect_nocs_mask_values(nocs_fixed, mask_binary)
# normalise nocs
nocs_norm = np.array(nocs_clean, dtype=np.float32) / 255.0


# Back-project depth points | image plane (u,v,Z) -> camera coordinate system (X,Y,Z))
pts, idxs = utils.backproject_opengl(depth, utils.get_intrinsics(), mask_binary)


### Load INFO
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
# get pose
pose_quat_wxyz = image_info["pose_quaternion_wxyz"]
location_xyz = image_info["location_xyz"]
RT = utils.convert_blender_pose_to_camera_object_pose(pose_quat_wxyz, location_xyz, metric_height, verbose=True)


# Get the nocs and metric points
if True:
    nocs_points = nocs_norm[idxs[0], idxs[1], :] - 0.5
else:
    gt_image_points = np.asarray(idxs).transpose().astype(np.float32)
    gt_image_points[:,[0, 1]] = gt_image_points[:,[1, 0]]
    nocs_points = nocs[gt_image_points[0],gt_image_points[1],:] - 0.5

gt_scale_factor = utils.get_space_dag(metric_width, metric_height, metric_depth, scale=1000, verbose=True)
metric_points = nocs_points * gt_scale_factor # in mm


#### Visualise in Open3D

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

# if __name__ == '__main__':
    