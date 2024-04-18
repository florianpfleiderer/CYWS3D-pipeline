# Created on Thu Apr 18 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
utility function for pointcloud annotation
"""

import copy
import open3d as o3d
import cv2
import numpy as np
try:
    from src.annotation_pipeline.projection import Intrinsic  
except ImportError:
    from projection import Intrinsic  # fallback for direct script execution


def read_annotation(path_to_annotation: str) -> dict:
    ''' Read annotation file and return dictionary '''
    with open (path_to_annotation, "r", encoding="utf-8") as f:
        anno = f.readlines()
        annotation_dictionary = {}
        for line in anno:
            line = line.split()
            obj_name = line[0]
            indices = line[1:-1]
            indices = [int(i) for i in indices]
            annotation_dictionary[obj_name] = indices
    return annotation_dictionary


def annotate_pcd(input_pcd: o3d.geometry.PointCloud, \
                 path_to_annotation: str, \
                 gt_colour: np.array = np.array([0.1, 0.9, 0.1])) \
                     -> (o3d.geometry.PointCloud, dict):
    ''' Annotate point cloud with indices from annotation file '''
    annotated_pcd = copy.deepcopy(input_pcd)
    with open (path_to_annotation, "r", encoding="utf-8") as f:
        anno = f.readlines()
        annotation_dictionary = {}
        for line in anno:
            line = line.split()
            obj_name = line[0]
            indices = line[1:-1]
            indices = [int(i) for i in indices]
            annotation_dictionary[obj_name] = indices

    for _, indices in annotation_dictionary.items():
        color = gt_colour
        for i in indices:
            annotated_pcd.colors[i] = color
    return annotated_pcd, annotation_dictionary


def extract_3d_bboxes(input_pcd: o3d.geometry.PointCloud, \
                        annotation_dictionary: dict, \
                        result=False) -> list:
    ''' Draw bounding boxes around annotated objects 
    
    Args:
        pcd: point cloud
        annotation_dictionary: dictionary of annotations

    Returns:
        list of bounding boxes
    '''
    bboxes = []
    for obj in annotation_dictionary:
        indices = annotation_dictionary[obj]
        points = np.asarray(input_pcd.points)[indices]
        bbox = o3d.geometry.AxisAlignedBoundingBox.\
            create_from_points(o3d.utility.Vector3dVector(points))
        bboxes.append(bbox)
        if result:
            o3d.visualization.draw_geometries([input_pcd, bbox])
    return bboxes


def draw_2d_bboxes(pixel_coordinates: tuple, \
                   points_color: np.array, \
                   gt_coordinates: tuple, intrinsics: Intrinsic):
    ''' Draw 2D bounding boxes on image 
    
    Args:
        pixel_coordinates: tuple of u, v coordinates
        points_color: np.array of colors
        gt_coordinates: tuple of gt_u, gt_v coordinates
        intrinsics: object of instance Intrinsic
        
    Returns:
        cv2 image with bounding boxes
    '''
    u_coords, v_coords = pixel_coordinates
    gt_u, gt_v = gt_coordinates

    assert len(gt_u) == len(gt_v) != 0, "gt_object not in fov"
    assert isinstance(intrinsics ,Intrinsic), f"intrinsics must be instance of Intrinsic Class,\
                                                but is {type(intrinsics)}"

    image = np.zeros((intrinsics.height, intrinsics.width, 3), dtype=np.uint8)
    points_color = points_color * 255
    # Fill the image array with your colors at the corresponding u, v coordinates
    for u, v, color in zip(u_coords, v_coords, points_color):
        image[int(v), int(u)] = color

    # if image.shape[2] == 3:  # Convert RGB to BGR for OpenCV if necessary
    #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    top_left = (min(gt_u), min(gt_v))
    bottom_right = (max(gt_u), max(gt_v))

    # Define the color and thickness of the bounding box
    color = (0, 255, 0)
    thickness = 2

    cv2.rectangle(image, top_left, bottom_right, color, thickness)

    return cv2.flip(image, 1)


def draw_2d_bboxes_on_img(image_path: str, gt_u, gt_v):
    ''' Draw 2D bounding boxes on image

    Args:
        image_path: path to image
        gt_u: list of x coordinates for ground truth
        gt_v: list of y coordinates for ground truth

    Returns:
        image with bounding boxes
    '''

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)

    min_u, max_u = min(gt_u), max(gt_u)
    min_v, max_v = min(gt_v), max(gt_v)

    top_left = (min_u, min_v)
    bottom_right = (max_u, max_v)

    # Define the color and thickness of the bounding box
    color = (0, 255, 0)  # Green for visibility
    thickness = 2  # Thickness of the box lines

    # Draw the rectangle
    cv2.rectangle(image, top_left, bottom_right, color, thickness)

    return cv2.flip(image, 1)


if __name__ == "__main__":
    PCD_NAME: str = "../../data/annotation/office/scene4/merged_plane_clouds_ds002.pcd"
    # PCD_NAME: str = "./testmodel/model1.pcd"
    ANNO_NAME: str = "../../data/annotation/office/scene4/merged_plane_clouds_ds002_GT.anno"

    pcd = o3d.io.read_point_cloud(PCD_NAME)
    print(pcd)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    pcd, annotation_dict = annotate_pcd(pcd, ANNO_NAME)

    o3d.visualization.draw_geometries([pcd, mesh_frame])
