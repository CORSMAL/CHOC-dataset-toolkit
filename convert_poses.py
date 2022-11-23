#!/usr/bin/env python
#
# Example code for converting poses in info.json to camera-to-object poses.
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

import numpy as np
import os
import json
import argparse
import utils

# Parsing arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--choc_dir', type=str, help="Path to the CHOC dataset.")
parser.add_argument('--image_index', type=str, help="Image index in 6-digit string format.", default="000001")
args = parser.parse_args()

# Get the pose from the JSON file of this image
image_info_path = os.path.join(args.choc_dir, "mixed-reality", "annotations", utils.image_index_to_batch_folder(args.image_index), "{}.json".format(args.image_index))
with open(image_info_path, 'r') as f:
    image_info = json.load(f)
object_id = image_info["object_id"]
pose_quat_wxyz = image_info["pose_quaternion_wxyz"]
location_xyz = image_info["location_xyz"]

print("Pose (Quaternion WXYZ):", pose_quat_wxyz)
print("Location (XYZ):", location_xyz)

# Get the height of the object in this image
object_datastructure_path = os.path.join(args.choc_dir, "mixed-reality", "extra", "object_models", "object_datastructure.json")
with open(object_datastructure_path, 'r') as f:
    objects_info = json.load(f)
metric_mm_height = objects_info["objects"][object_id]["height"]

# Convert the pose
converted_pose = utils.convert_blender_pose_to_camera_object_pose(pose_quat_wxyz, location_xyz, metric_mm_height, verbose=True)