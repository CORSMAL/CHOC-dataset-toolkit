#!/usr/bin/env python
#
# Utility functions for the CHOC dataset toolkit.
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
np.set_printoptions(suppress=True) # do not with print scientific notation
import math
import sys
from scipy.spatial.transform import Rotation as R
import utils
import cv2

def fix_background_nocs(nocs_img):
	"""
	Inputs NOCS [h,w,3], [0-255], uint8. Incorrect background values. [13,13,13], [14,14,14]
	Output NOCS [h,w,3], [0-255], uint8. Black background. [0,0,0]
	"""
	# Convert to grayscale
	nocs_gray = cv2.cvtColor(nocs_img, cv2.COLOR_RGB2GRAY)
	# Threshold the image using OTSU
	rect, nocs_binarized = cv2.threshold(nocs_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	# NOCS binary
	nocs_binary = nocs_binarized/255.0
	nocs_bin_stack = np.stack([nocs_binary, nocs_binary, nocs_binary], axis=2)
	nocs_processed = nocs_img * nocs_bin_stack
	nocs_processed = nocs_processed.astype(np.uint8)
	return nocs_processed

def convert_blender_pose_to_camera_object_pose(pose_quat_wxyz, location_xyz, metric_mm_height, verbose=False):
	"""
	The annotated pose in the info.json files in CHOC (mixed-reality part) is the pose that places the object in the Blender environment.
	It is not the camera-object pose. Let's make it so.

	Inputs
	------
	pose_quat_wxyz: 4-D float array
		pose in quaternion WXYZ order
	location_xyz: 3-D float array
		location in meter XYZ order
	metric_mm_height: float
		height of the object in millimeters

	Outputs
	-------
	blender_RT_corrected: [4,4 matrix]
		a 4x4 transformation matrix
	"""
	
	# Convert to the order that is needed by Scipy Rotation library
	pose_quat_xyzw = [pose_quat_wxyz[1], pose_quat_wxyz[2], pose_quat_wxyz[3], pose_quat_wxyz[0]]

	# Load rotation object
	r = R.from_quat(pose_quat_xyzw)
	rot_matrix = r.as_matrix() # for older versions of scipy, use: r.as_dcm()
	blender_RT = np.zeros((4,4))  # placeholder
	blender_RT[:3,:3] = rot_matrix
	blender_RT[:3, 3] = np.asarray(location_xyz) * 1000 # multiply by 1000 - because we convert meter to millimeter!
	blender_RT[3,3] = 1
	
	# Add the 90 degrees camera rotation (because this was done during rendering)
	rot_90_camera = np.zeros((4,4))
	rot_90_x = get_rotation_about_axis(theta=math.radians(-90), axis="X")
	rot_90_camera[:3,:3] = rot_90_x
	rot_90_camera[3,3] = 1
	
	# Translate the object, so that the centroid of the object is at the origin
	correcting_RT = [[1,0,0,0],
					 [0,1,0,0],
					 [0,0,1,(metric_mm_height*1000)/2],
					 [0,0,0,1]]
	blender_RT_corrected = blender_RT @ correcting_RT
	blender_RT_corrected = rot_90_camera @ blender_RT_corrected 
	
	# Print result
	if verbose:
		print("\nPose for blender:")
		print(blender_RT)

		print("\nPose between camera and object:")
		print(blender_RT_corrected)
	
	return blender_RT_corrected

def opengl_to_opencv(RT):
	"""
	Converts a [4,4] transformation matrix in OpenGL coordinate system, 
	 to a [4,4] transformation matrix in OpenCV coordinate system.

	https://stackoverflow.com/questions/44375149/opencv-to-opengl-coordinate-system-transform
	"""
	rot_180_x = get_rotation_about_axis(math.radians(180), axis="X")
	RT_180_x = np.zeros((4,4))
	RT_180_x[:3,:3] = rot_180_x
	RT_180_x[3, 3] = 1
	return RT_180_x @ RT # First go to OpenCV format, then apply RT

def get_avg_scale_factor(objects_info, object_id, verbose=False):
	"""
	Inputs
	------
	objects_info : dict
		loaded from objects_datastructure.json
	object_id : int
		CHOC object ID

	Outputs
	-------
	avg_scale_factor: float
		The average scaling/normalizing factor of the category where object_id belongs to
	"""
	avg_scale_factor = objects_info["categories"][object_id]["average_train_scale_factor"]
	if verbose:
		print("avg_scale_factor:", avg_scale_factor)
	return avg_scale_factor

def get_gt_scale_factor(objects_info, object_id):
	"""
	Inputs
	------
	objects_info : dict
		loaded from objects_datastructure.json
	object_id : int
		CHOC object ID

	Outputs
	-------
	gt_scale_factor : float
		the ground truth scaling/normalizing factor of the category where object_id belongs to
	"""
	width_mm = objects_info["objects"][object_id]["width"]  # mm means millimeter
	height_mm = objects_info["objects"][object_id]["height"] 
	depth_mm = objects_info["objects"][object_id]["depth"] 
	gt_scale_factor = get_space_dag(width_mm, height_mm, depth_mm)
	return gt_scale_factor

def remove_duplicates(pts, image_pts):
	"""
	Removes duplicates from 3D points, and corresponding image 2D points.

	Inputs
	------
	pts : [N, 3]
		3D points
	image_pts : [N, 2]
		corresponding 2D pixels
	
	Outputs
	------
	pts : [M, 3]
		3D points (duplicates removed)
	image_pts : [M, 2]
		corresponding 2D pixels (duplicates removed)
	"""
	unique, indices, indices_inverse, counts = np.unique(pts, axis=0, return_index=True, return_inverse=True, return_counts=True)
	indices = indices[2:]
	indices.sort()
	cat = np.hstack([pts, image_pts])
	return cat[indices,:3], cat[indices,3:]

def run_umeyama(raw_coord, raw_depth, mask, image_index, verbose=False):
	"""
	Runs Umeyama similarity transform on NOCS and corresponding DEPTH points.

	nocs format: rgb, [0-1]
	depth format:
	mask format:

	Returns transformation matrix [4,4] in OpenGL format. Pose is for metric object.
	"""
	success_flag = False
	try:
		# Get depth into pointcloud
		pts, idxs = backproject_opengl(raw_depth, get_intrinsics(), mask)
		# Get corresponding nocs points
		coord_pts = raw_coord[idxs[0], idxs[1], :] - 0.5

		# Compute pose and scale scalar
		umey_scale_factors, rotation, translation, outtransform = estimateSimilarityTransform(coord_pts, pts, False)
		if verbose:
			print("umeyama scale factor:", umey_scale_factors)	
		
		# Make the 4x4 matrix
		umeyama_RT = np.zeros((4, 4), dtype=np.float32) 
		with_scale = False # don't making scaling part of the transformation
		if with_scale:
			umeyama_RT[:3, :3] = np.diag(umey_scale_factors) / 1000 @ rotation.transpose() # @ = matrix multiplication
		else:
			umeyama_RT[:3, :3] = rotation.transpose()
		umeyama_RT[:3, 3] = translation # NOTE: do we need scaling? # / 1000 # meters
		umeyama_RT[3, 3] = 1
		success_flag=True

	except Exception as e:
		message = '[ Error ] aligning instance {} in {} fails. Message: {}.'.format("object", image_index, str(e))
		print(message)
		umeyama_RT = np.zeros((4, 4), dtype=np.float32) 
		umey_scale_factors = [0,0,0]


	return umeyama_RT, umey_scale_factors, success_flag

def run_epnp(object_points, image_points, verbose=False):
	"""
	Runs EPnP and RANSAC on the metric object (nocs*scale) and corresponding image points.
	Returns the pose in OpenGL format. 
	The pose is metric scale, I.E. it is transforming the objects points.
	
	Inputs:
		object_points [N,3]
		image_points [N,2]
	
	Outputs:
		transformation matrix [4,4]

	Other option:
	 - solvePnPGeneric (no RANSAC)
	"""
	retval, rvec, tvec, reprojectionError = cv2.solvePnPRansac(objectPoints=object_points, 
															   imagePoints=image_points,
															   cameraMatrix=get_intrinsics(), 
															   distCoeffs=None,
															   useExtrinsicGuess = False,
															   iterationsCount=100,
															   reprojectionError = 2.0,
															   confidence = 0.99,
															   flags=cv2.SOLVEPNP_EPNP) #cv2.SOLVEPNP_SQPNP 
	#rvec = rvecs[0]
	#tvec = tvecs[0]

	# Convert to a 4x4 Matrix
	epnp_RT = np.zeros((4, 4), dtype=np.float32) 
	#tvec *= gt_scale_factor
	pred_R = cv2.Rodrigues(rvec)[0]
	epnp_RT[:3, :3] = pred_R   #.transpose() # np.diag([scale_factor, scale_factor, scale_factor]) @ pred_R.transpose()
	epnp_RT[:3, 3] = tvec[:,0] # / 1000
	epnp_RT[3, 3] = 1
	# Add -Y -Z, which is 180 rotation about X
	rot_180_camera = np.zeros((4,4))
	rot_180_X = get_rotation_about_axis(theta=math.radians(-180), axis="X")
	rot_180_camera[:3,:3] = rot_180_X
	rot_180_camera[3,3] = 1
	epnp_RT = rot_180_camera @ epnp_RT

	if verbose:
		print("===PnP results===")
		print('Number of solutions = {}'.format(len(rvecs)))
		print('Rvec = {}, tvec = {}'.format(rvec, tvec))
		print('Reprojection error = {}'.format(reprojectionError))
		print("\nEPNP pose:")
		print(epnp_RT)

	return epnp_RT

def map_pixelValue_to_classId(pixelValue):
	"""
	Maps pixel values in the mask to the corresponding class id.
	"""
	classId = None
	if pixelValue == 0:
		return 0
	elif pixelValue == 50:
		return 1
	elif pixelValue == 100:
		return 2
	elif pixelValue == 150:
		return 3
	elif pixelValue == 200:
		return 4
	raise Exception("Pixel value should be one of: [0,50,100,150,200]. You gave as input:", pixelValue)

def intersect_nocs_mask_values(nocs, mask):
	"""
	The mask and nocs are not the same, i.e. some pixel locations are non-zero in nocs and zero in mask, and vice versa.
	This means that if we mask the nocs, we get some background values in the NOCS points. Let's avoid this.

	nocs [h,w,3] in range [0-255]
	mask [h,w,3] in range [0-1]
	"""
	#placeholder
	nocs_clean = nocs.copy()

	# Get all image points where mask AND nocs is nonzeros
	s = np.sum(nocs_clean,axis=2)
	x,y = np.where((mask!=0) & (s!=0))
	xy_image = np.where((mask!=0) & (s!=0), 1, 0)

	# Convert to 3 channels
	xy_image3 = np.stack([xy_image, xy_image, xy_image], axis=2)
	
	# Apply the new mask
	nocs_cleaned = nocs_clean * xy_image3
	nocs_intersected = nocs_cleaned.astype('uint8')
	mask_intersected = mask  * xy_image

	points = np.where(xy_image != 0)

	return nocs_intersected, mask_intersected, points

def backproject_opengl(depth, intrinsics, instance_mask):
	"""Back-projecting points, i.e. 2D pixels + depth + intrinsics --> 3D coordinates
	
	We apply the instance mask to the 2D depth image, because we only want the depth points of the object of interest.
	We then backproject these 2D pixels (U,V) to 3D coordinates (X,Y,Z) using the camera intrinsics and depth (Z). 
	"""

	# Compute the (multiplicative) inverse of a matrix.
	intrinsics_inv = np.linalg.inv(intrinsics)
	# Get shape of the depth image
	image_shape = depth.shape
	width = image_shape[1]
	height = image_shape[0]
	# Returns evenly spaced values, default = 1. This case: return x = [0,1,2,...,width]
	x = np.arange(width) 
	y = np.arange(height)
	# Get binary mask where values are positive if both depth and instance mask are 1
	#non_zero_mask = np.logical_and(depth > 0, depth < 5000)
	non_zero_mask = (depth > 0)
	final_instance_mask = np.logical_and(instance_mask, non_zero_mask) 
	# Get all coordinates of this mask where values are positive
	idxs = np.where(final_instance_mask)
	grid = np.array([idxs[1], idxs[0]]) # Shape = (2,N) where N is number of points

	# Add a Z-coordinate (all 1s)
	N = grid.shape[1]
	ones = np.ones([1, N])
	uv_grid = np.concatenate((grid, ones), axis=0) # Shape = (3, N) where N is number of points
	xyz = intrinsics_inv @ uv_grid # (3, N)
	xyz = np.transpose(xyz) # (N, 3)

	z = depth[idxs[0], idxs[1]] 

	pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
	
	# To OpenGL
	pts[:, 1] = -pts[:, 1]
	pts[:, 2] = -pts[:, 2]

	return pts, idxs # 3d point coordinates, 2d pixel coordinates

def get_lines():
	"""
	Lines to draw a 3D bounding box.
	"""
	
	# Draw the BOUNDING BOX
	lines = [
		# Ground rectangle
		[1,3],
		[3,7],
		[5,7],
		[5,1],

		# Pillars
		[0,1],
		[2,3],
		[4,5],
		[6,7],

		# Top rectangle
		[0,2],
		[2,6],
		[4,6],
		[4,0]
	]
	return lines
	
def get_class_names():
	return ['BG',       #0
			'box',      #1
			'non-stem', #2
			'stem',     #3
			'person']   #4

def mask_remove_hand_and_binarize(gt_mask):
	"""
	Takes the raw CHOC (mixed-reality) mask. Removes the hand. Converts to binary.
	"""
	new_mask = gt_mask.copy()
	new_mask[gt_mask == 200] = 0
	pixelValue = np.unique(new_mask)[1]
	new_mask[new_mask == pixelValue] = 1
	return new_mask

def get_intrinsics():
	"""
	Camera intrinsics for the CHOC mixed-reality dataset.
	"""
	fx = 605.408875 # pixels
	fy = 604.509033 # pixels
	cx = 320        # cx = 321.112396 # pixels
	cy = 240        # cy = 251.401978 # pixels
	intrinsics = np.array([[fx, 0, cx], [0., fy, cy], [0., 0., 1.]])
	return intrinsics

def get_intrinsics_ccm():
	
	intrinsics = np.asarray([[923, 0.,  640], 			
							 [0.,  923, 360], 			
							 [0.,  0.,  1.]])

	return intrinsics

def image_index_to_batch_folder(image_index):
	"""
	This will give you the folder of the batch where the CHOC mixed-reality image is.

	Inputs
	------
	image_index : 6-digit string
		an image from the CHOC mixed-reality dataset
	
	Outputs
	-------
	foldername : string
		name of the corresponding batch folder
	"""
	
	# Remove leading zeros
	string = str(image_index).lstrip("0")
	x = int(string)

	if ((x % 1000) == 0):
		# get batch1 and batch2 
		b1 = x-999
		b2 = x
		foldername = "b_{:06d}_{:06d}".format(b1,b2)
	else:
		y = x - (x % 1000)
		# get batch1 and batch2 
		b1 = y+1
		b2 = y+1000
		foldername = "b_{:06d}_{:06d}".format(b1,b2)
	return foldername

def get_space_dag(w,h,d, scale=1, verbose=True):
	"""
	Calculates the Space Diagonal of a 3D box, using the Pythagoras Theorem.
	https://en.wikipedia.org/wiki/Space_diagonal

	Inputs
	------
	w : float
		width of the box
	h : float
		height of the box
	d : float
		depth of the box
	
	Outputs
	-------
	space_dag : float
		The space diagonal of the bounding box.
	"""
	space_dag = math.sqrt( math.pow(w,2) + math.pow(h,2) + math.pow(d,2) )
	if verbose:
		print("Object dimensions: ({:.2f} m, {:.2f} m, {:.2f} m)".format(w,h,d))
		print("Space diagonal:", space_dag, "m")
	return space_dag * scale

def get_rotation_about_axis(theta, axis=None):
	"""
	Gives you the [3,3] rotation matrix around a certain axis, with theta radians.
	
	Inputs
	------
	theta : float
		angle in radians
	axis : string
		X, Y or Z depending on which axis you want to rotate around

	Outputs
	-------
	mat : [3,3]
		rotation matrix
	"""
	if axis == "X":
		mat = np.array( [ [1, 0,              0            ],
						  [0, np.cos(theta), -np.sin(theta)],
						  [0, np.sin(theta),  np.cos(theta)]])

	elif axis == "Y":
		mat = np.array( [ [ np.cos(theta), 0, np.sin(theta)],
						  [ 0,             1, 0            ],
						  [-np.sin(theta), 0, np.cos(theta)]])
	
	elif axis == "Z":
		mat = np.array( [ [np.cos(theta), -np.sin(theta), 0],
						  [np.sin(theta),  np.cos(theta), 0],
						  [0,              0,             1]])
	else:
		raise Exception("Unknown axis:", axis)

	return mat

def transform_coordinates_3d(coordinates, RT):
    """
	Transforms a set of 3D points, with a transformation matrix 'RT'.
	From: https://github.com/hughw19/NOCS_CVPR2019/blob/78a31c2026a954add1a2711286ff45ce1603b8ab/utils.py#L668

    Input: 
        coordinates: [3, N]
        RT: [4, 4]
    Return 
        new_coordinates: [3, N]

    """
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates

def compute_RT_degree_cm_symmetry(RT_1, RT_2, class_name):
	"""
	Compute the difference in rotation and translation between two transformation matrices.
	Code mostly from: https://github.com/hughw19/NOCS_CVPR2019/blob/master/utils.py#L338
	
	Inputs
	------
	RT_1 : [4, 4]
		homogeneous affine transformation; translation is in millimeter
	RT_2 : [4, 4]
		homogeneous affine transformation; translation is in millimeter
	class_name : string
		name of the CHOC category
	
	Outputs
	-------
	theta : float
		difference in 3D angle between RT_1 and RT_2 (degrees)
	shift : float
		L2 difference of T in centimeter
	"""
	## make sure the last row is [0, 0, 0, 1]
	if RT_1 is None or RT_2 is None:
		return -1
	try:
		assert np.array_equal(RT_1[3, :], RT_2[3, :])
		assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
	except AssertionError:
		print(RT_1[3, :], RT_2[3, :])
		exit()

	R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
	T1 = RT_1[:3, 3]
	R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
	T2 = RT_2[:3, 3]

	# Compute theta for an object that is symmetric when rotating around Z-axis 
	if class_name in ['stem', 'non-stem']: 
		z = np.array([0, 0, 1])
		y1 = R1 @ z
		y2 = R2 @ z
		theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
	
	# Compute theta for an object that is symmetric when rotated 180 degrees around Z-axis 
	elif class_name in ['box']:
		z_180_RT = np.diag([-1.0, -1.0, 1.0])
		
		# Step 1a. Compute the difference in rotation between these two matrices
		R = R1 @ R2.transpose() 
		# Step 1b. Compute the difference in rotation between these two matrices, but rotated 180 degrees around Z-axis
		R_rot = R1 @ z_180_RT @ R2.transpose() 
		
		# Step 2. Compute the axis-angle (ω, θ) representation of R and R_rot using the following formula
		# We take the minimum rotational error, because we want the loss of a box to be equal to the same box but rotation 180 degrees around y-axis.
		theta = min(np.arccos((np.trace(R) - 1) / 2),
					np.arccos((np.trace(R_rot) - 1) / 2))
	
	# Compute theta for an object that has no symmetry
	else:
		R = R1 @ R2.transpose()
		theta = np.arccos((np.trace(R) - 1) / 2)
	
	theta *= 180 / np.pi # Radian to degrees
	shift = np.linalg.norm(T1 - T2) / 10 # Why divide by 10 ? Answer: to go from millimeter to centimeter
	result = np.array([theta, shift])
	return result

