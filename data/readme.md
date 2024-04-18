## Pipeline data
GH30_\<roomname\>...contains the selected RGB Frames from the original input data (rosbag files) </br>
ObChange............contains the Pointclouds and Ground Truth annotations </br>
predictions.........target folder for cyws3d inference results </br>
inference...........source folder for images; contains scripts for preparing the rgb frames and creating the yaml file needed for inference </br>
annotation..........temporary folder, contains only parts of ObChange </br>

### GH30
This Folder contains the selected Frames from the RGB and Depth Stream of ObChange.

#### Naming sceme:
img\<imagenumber\>\_s\<scene\>\_\<L / R\>_00_crop.png <br />
img\<imagenumber\>\_s\<scene\>\_\<L / R\>_00_crop_depth.png

### ObChange
Point Clouds of reconstructed surfaces are inside teh folder. </br>
Folder '0' contains the pcd file for the whole scene.