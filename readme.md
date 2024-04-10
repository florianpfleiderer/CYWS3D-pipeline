# [The Change You Want to See (Now in 3D)](#)

[[Project Page]](#) [[arXiv]](https://arxiv.org/abs/2308.10417)

In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) 2023

[Ragav Sachdeva](https://ragavsachdeva.github.io/), [Andrew Zisserman](https://scholar.google.com/citations?hl=en&user=UZ5wscMAAAAJ)

![results](demo_data/pred.gif)

## Installation

**Clone the repository**

```
git clone --recursive git@github.com:ragavsachdeva/CYWS-3D.git
```

**Install depedencies**

```
docker build -t cyws3d:latest .
```

**Notes**

Showing images with cv2 inside docker currently does not work due to the following error:
```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/miniconda3/envs/cyws3d-pipeline/lib/python3.9/site-packages/cv2/qt/plugins" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: xcb.

Aborted (core dumped)
```


## Datasets

KC-3D: `wget https://thor.robots.ox.ac.uk/cyws-3d/kc3d.tar`

RC-3D: `wget https://thor.robots.ox.ac.uk/cyws-3d/rc3d.tar`


## Pre-trained model

```
wget https://thor.robots.ox.ac.uk/cyws-3d/cyws-3d.ckpt.gz
gzip -d cyws-3d.ckpt.gz
```

## Example Usage

Please try running:

`python inference.py --load_weights_from ./cyws-3d.ckpt`

This should perform a batched inference on a set of example image pairs under various settings (see [this file](demo_data/input_metadata.yml)).

For the annotation pipeline try:

`annotate.py`

This should run scripts/annotate.py as the script gets installed through setup.py.

## Citation

```
@InProceedings{Sachdeva_ICCVW_2023,
    title = {The Change You Want to See (Now in 3D)},
    author = {Sachdeva, Ragav and Zisserman, Andrew},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year = {2023},
}
```
