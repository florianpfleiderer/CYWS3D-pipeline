## Pipeline data
GH30_\<roomname\>...contains the selected RGB Frames from the original input data (rosbag files) </br>
ObChange............contains the Pointclouds and Ground Truth annotations </br>
inference...........source folder for demo and test images; contains old scripts, not used anymore </br>
annotation..........temporary folder, contains only parts of ObChange </br>

### GH30
This Folder contains the selected Frames from the RGB and Depth Stream of ObChange.

#### Naming sceme:
<rbg_ros_topic\>-<timestamp\>.png <br />
<depth_ros_topic\>-<timestamp\>.png <br />

### ObChange
Point Clouds of reconstructed surfaces are inside the folder. </br>
Folder '0' contains the pcd file for the whole scene.