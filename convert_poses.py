"""
Example code for converting poses in info.json to camera-to-object poses.
"""
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

# Get the height of the object in this image
object_datastructure_path = os.path.join(args.choc_dir, "mixed-reality", "extra", "object_models", "object_datastructure.json")
with open(object_datastructure_path, 'r') as f:
    objects_info = json.load(f)
metric_mm_height = objects_info["objects"][object_id]["height"]

# Convert the pose
converted_pose = utils.convert_blender_pose_to_camera_object_pose(pose_quat_wxyz, location_xyz, metric_mm_height, verbose=True)