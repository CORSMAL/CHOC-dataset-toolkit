# CHOC-dataset-toolkit
Toolkit for for the CORSMAL Hand-Occluded Containers (CHOC) dataset with codes to inspect the data in 3D, clean the NOCS images, and convert 6D object poses.

## TODO
- [ ] Add code sample to inspect data in 3D
- [ ] Add code sample to clean the NOCS images
- [ ] Add code sample to convert Blender poses to camera-object poses
- [ ] Add instructions to load the GraspIt! World files

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
