# Toolkit for the CHOC dataset

Toolkit for the CORSMAL Hand-Occluded Containers (CHOC) dataset with codes to inspect the 3D data, clean the NOCS images, convert 6D object poses, instructions, and other utility functions. 

[[dataset](https://zenodo.org/record/5085801#.Y3zGQ9LP2V4)]
[[webpage](https://corsmal.eecs.qmul.ac.uk/pose.html)]
[[arxiv pre-print](https://arxiv.org/abs/2211.10470)]

## Table of contents
1. [Installation](#requirements)
2. [Running sample codes](#running)
   1. [Inspect the data](#inspect)
   2. [Clean the NOCS backgrounds](#clean)
   3. [Convert poses](#convert)
3. [Load GraspIt! worlds](#instructions)
4. [Enquiries](#enquiries)
6. [License](#license)

## Installation <a name="requirements"></a>

The toolkit was implemented and tested with the following requirements on a machine with Ubuntu 18.04

Requirements:
- Anaconda 4.13.0
- Python 3.9
- SciPy 1.9.3
- Open3D 0.16.0
- NumPy 1.23.5
- OpenCV 4.5.5.64

Use the following commands to install the toolkit within an Anaconda environment:
```
conda create -n CHOC-toolkit-env python=3.9
conda activate CHOC-toolkit-env
pip install -r requirements.txt
```

## Running sample codes <a name="running"></a>

Here, we will explain how to inspect the data in 3D, clean the NOCS backgrounds, and convert the annotated poses into camera2object poses.
We will use "000251" in the CHOC mixed-reality data as example, which you can also find in this repository in the _sample_ folder.

<details>
<summary> Show images for 000251</summary>

<br>

  RGB                       |  NOCS                     |  Mask                     |  Depth
:--------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![RGB](sample/CHOC/mixed-reality/rgb/b_000001_001000/000251.png) |![NOCS](sample/CHOC/mixed-reality/nocs/b_000001_001000/000251.png)|![Mask](sample/CHOC/mixed-reality/mask/b_000001_001000/000251.png)|![Depth](images/depth.png)

</details>

### Inspecting the data <a name="inspect"></a>

We provide sample code to visualise the data in 3D.
```
python main.py --choc_dir <path_to_choc> --image_index 000251 --operation inspect
```

<details>
<summary> Show result for 000251</summary>

<br>

Here we visualise the un-normalised NOCS-points in green; the depth points in blue; the un-normalised NOCS points transformed using the converted pose in red. They are all visualised in the camera coordinate system (OpenGL convention).

  Object                      |  Depth, Annotation        |  Both                     
:----------------------------:|:-------------------------:|:-------------------------:
![Metric object points](images/object.png) |![Depth; Transformed object](images/depth_and_transformed_object.png)|![Both](images/both.png)

</details>


### Clean the NOCS backgrounds <a name="clean"></a>

Due to an issue in the rendering process, the background pixels of the NOCS images are not truly black, i.e. [0,0,0]. We provide sample code to fix this.

```
python main.py --choc_dir <path_to_choc> --image_index 000251 --operation fix_nocs
```
<details>
<summary> Show before and after for 000251</summary>

<br>

NOTE: It seems that the background pixels are always [13,13,13] or [14,14,14], but we haven't verified that.
Therefore we choose to segment the foreground from background using Otsu's algorithm:
We use https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html#:~:text=Otsu's%20Binarization,determines%20it%20automatically.

Otsu works well, since there is a clear separation between foreground/background, but we found it is not perfect. I.e. sometimes, 
it will remove some NOCS pixels.

In this visualisation of the result, we zoom in on the pixels. Note how the background pixels were [13,13,13] or [14,14,14] before; and [0,0,0] after using Otsu's method.


  Before                    |  After
:--------------------------:|:-------------------------:
![Before processing](images/nocs_before.png) |![After processing](images/nocs_after.png)

</details>

### Convert the poses <a name="convert"></a>

Here's an example of the annotated file for image "000251".
```
{
    "background_id": "000016.png",
    "flip_box": false,
    "grasp_id": 0,
    "location_xyz": [
        -0.1323072761297226,
        1.0679999589920044,
        -0.029042737558484077
    ],
    "object_id": 29,
    "pose_quaternion_wxyz": [
        0.9741891026496887,
        0.16082336008548737,
        0.15678565204143524,
        -0.022577593103051186
    ],
    "occlusion_ratio": 37.99846625766871
}
```
The _location\_xyz_ and _pose\_quaternion\_wxyz_ is the pose that was used to place the object inside the blender environment. It is NOT the camera-object pose. To convert the pose, you can do as follows:
```
python main.py --choc_dir <path_to_choc> --image_index <image_index_string> --operation convert_pose
```

For image_index "000251" the result will be:
```
Pose for blender:
[[   0.94981703    0.09441928    0.29821572 -132.30727613]
 [   0.0064399     0.9472522    -0.3204244  1067.99995899]
 [  -0.31273974    0.30626503    0.89910822  -29.04273756]
 [   0.            0.            0.            1.        ]]

Pose between camera and object:
[[    0.94981703     0.09441928     0.29821572  -114.63542574]
 [   -0.31273974     0.30626503     0.89910822    24.237169  ]
 [   -0.0064399     -0.9472522      0.3204244  -1049.01205329]
 [    0.             0.             0.             1.        ]]
```
Pose for blender is simply _location\_xyz_ and _pose\_quaternion\_wxyz_ converted into a 4x4 transformation matrix.
Pose between camera and object is the 4x4 transformation matrix between the camera and object.

## Loading GraspIt! world files <a name="instructions"></a>

1. Install ROS Melodic (or another version): http://wiki.ros.org/melodic/Installation/Ubuntu
2. Install GraspIt!
   * https://graspit-simulator.github.io/build/html/installation_linux.html
   * https://github.com/graspit-simulator/graspit_interface

3. Install ManoGrasp: https://github.com/ikalevatykh/mano_grasp (see ‘Install’ and ‘Model’ steps)
4. Open GraspIt! via terminal
```
$ source <your_graspit_ros_workspace>/devel/setup.bash
$ roslaunch graspit_interface graspit_interface.launch
```
5. Convert object files from .glb to .off using Open3D (sample code):

```python
import open3d as o3d

# Load .glb file
mesh = o3d.io.read_triangle_mesh(<path_to_input_glb_file>)

# Save as .off file
o3d.io.write_triangle_mesh(<path_to_output_off_file>, mesh)
```
All object .off files should be placed inside your GraspIt! workspace > objects > object_models

6. Load the .xml GraspIt! world (hand and object) from File > Import World (e.g. right_hand_bottom_box_01.xml) 


## Enquiries <a name="enquiries"></a>

For any questions, please open an issue on this repository, or send an email to corsmal-challenge@qmul.ac.uk.


## License <a name="license"></a>

This work is licensed under the MIT License. To view a copy of this license, see [LICENSE](LICENSE).
