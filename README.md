# CHOC-dataset-toolkit
Toolkit for the CORSMAL Hand-Occluded Containers (CHOC) dataset with codes to inspect the 3D data, clean the NOCS images, convert 6D object poses, instructions, and other utility functions. You can download the CHOC dataset [here](https://zenodo.org/record/5085801#.Y3zGQ9LP2V4), and read the corresponding paper [here](https://arxiv.org/abs/2211.10470).

[Webpage](https://corsmal.eecs.qmul.ac.uk/pose.html)

### Install requirements

- SciPy
- Open3D
- NumPy
- OpenCV

This code has been tested with python 3.9, but should work with other versions. You can make and activate a conda environment:
```
conda create -n CHOC-toolkit-env python=3.9
conda activate CHOC-toolkit-env

```

You can install the dependencies as follows:
```
pip install -r requirements.txt
```

### Running the sample codes

#### Inspecting the data
```
python inspect_data.py --choc_dir <path_to_choc>
```

#### Clean the NOCS backgrounds
```
python fix_nocs.py --choc_dir <path_to_choc>
```

  Before                    |  After
:--------------------------:|:-------------------------:
![Before processing](images/nocs_before.png) |![After processing](images/nocs_after.png)

#### Convert the poses
```
python convert_poses.py --choc_dir <path_to_choc>
```

### Other instructions

<details>
<summary> Instructions to load the GraspIt! world files</summary>

<br>
  
#### 
1. Install ROS Melodic (or another version)
 * Follow: http://wiki.ros.org/melodic/Installation/Ubuntu

2. Install GraspIt!
 * First follow: https://graspit-simulator.github.io/build/html/installation_linux.html
 * Then follow: https://github.com/graspit-simulator/graspit_interface

3. Install ManoGrasp
 * Follow the steps ‘Install’ and ‘Model’ in https://github.com/ikalevatykh/mano_grasp

4. Open GraspIt! via terminal
```
$ source <your_graspit_ros_workspace>/devel/setup.bash
$ roslaunch graspit_interface graspit_interface.launch
```

5. Convert object files from .glb to .off
 * Convert .glb files to .off. Here's a Python code sample:

```python
import open3d as o3d

# Load .glb file
mesh = o3d.io.read_triangle_mesh(<path_to_input_glb_file>)

# Save as .off file
o3d.io.write_triangle_mesh(<path_to_output_off_file>, mesh)
```

 * Put all object .off files inside your GraspIt! workspace > objects > object_models

6. Load our GraspIt! world to load the hand and object
 * File > Import World > Look for the .xml files in graspit_worlds (e.g. right_hand_bottom_box_01.xml) 
</details>

### Enquiries

For any questions, please open an Issue on this repository, or send an email to corsmal-challenge@qmul.ac.uk.

### License

This work is licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.