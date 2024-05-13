# CYWS3D Evaluation Pipeline on ObChange Dataset
The Goal is to perform a batched inference (one Batch per Room) on selected RGB-D
Frames taken from the original Input Data in the ObChange Dataset. The Pipeline is split into
3 Modules: 
1. Inference: takes the input images from data/GH30_\<roomname\> performs an inference and saves the images and bboxes in data/Gh30_\<roomname\>/predictions
2. Annotation: split into 2 Parts: 
    1. Extracting Transformations for selected images from the rosbag files (separate repo)
    2. Exract ground truth bounding boxes from data/ObChange/...
3. Evaluation: calculating the mAP
    1. note that the boundingboxes from the inference are in one file per room, sorted in rising order following scenename and timestamp in each scene (see input_metadata.yaml for the order)



## Installation

**Clone the repository**

```
git clone --recursive git@github.com:ragavsachdeva/CYWS-3D.git
```

**Install depedencies**

```
docker build -t cyws3d:latest .
```

Or start the .devcontainer using visual studio code (recommended way due to the postCreateCommand that is also run)

**Notes**

Showing images with cv2 inside docker currently does not work and produces the following error:
```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/miniconda3/envs/cyws3d-pipeline/lib/python3.9/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb.

Aborted (core dumped)
```


## Datasets

The Frames taken from the ObChange Dataset are found in data/GH30_\<roomname\>/...

## Pre-trained model

```
wget https://thor.robots.ox.ac.uk/cyws-3d/cyws-3d.ckpt.gz
gzip -d cyws-3d.ckpt.gz
```

## Example Usage
Make sure you are inside root folder.

After initializing the project with:
`pip install -e .`

start with annotation for the specific room:
`annotate.py --room <roomname>`

this annotates the dataset and draws boundingboxes for double checking in ground_truth folder into each scene.

next, create the metadata for inference: 
`create_inference_metadata.py --room <roomname> --depth <true/false> --transformations <true/false> --perspective <2d/3d>`

For inference please try running:
`inference.py --room <roomname>`

This should perform a batched inference on a set of example image pairs under various settings (see [this file](data/inference/demo_data/input_metadata.yml)).

The inference create the annotated predictions and saves the bboxes into GH30_*/predictions.

For evaluation: 
`evaluate.py --room <roomname>`

This saves the mAP file into predictions folder and some graphs. The mAP file can be loaded later for more data analysis.


