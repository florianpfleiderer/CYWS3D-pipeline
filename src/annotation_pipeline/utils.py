# Created on Thu Apr 18 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
utility function for pointcloud annotation
"""

import copy
import open3d as o3d
import cv2
import numpy as np
import yaml
try:
    from src.annotation_pipeline.projection import Intrinsic  
    from src.globals import GT_COLOR
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


def draw_image(pixel_coordinates: tuple, \
                   points_color: np.array, \
                   intrinsics: Intrinsic):
    ''' Draw image 
    
    Args:
        pixel_coordinates: tuple of u, v coordinates
        points_color: np.array of colors
        intrinsics: object of instance Intrinsic
        
    Returns:
        cv2 image with bounding boxes
    '''
    u_coords, v_coords = pixel_coordinates

    assert isinstance(intrinsics ,Intrinsic), f"intrinsics must be instance of Intrinsic Class,\
                                                but is {type(intrinsics)}"

    image = np.zeros((intrinsics.height, intrinsics.width, 3), dtype=np.uint8)
    points_color = points_color * 255
    # Fill the image array with your colors at the corresponding u, v coordinates
    for u, v, color in zip(u_coords, v_coords, points_color):
        image[int(v), int(u)] = color

    return image


def draw_2d_bboxes_on_img(image_file, bboxes: list):
    ''' Draw 2D bounding boxes on image

    Args:
        image_path: path to image
        bboxes: list of bounding boxes

    Returns:
        image with bounding boxes
    '''
    if isinstance(image_file, str):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image_file, np.ndarray):
        image = image_file
    else:
        raise ValueError("image_file must be a path to an image or a numpy array")

    bbox_color = GT_COLOR * 255
    bbox_thickness = 2 
    for box in bboxes:
        cv2.rectangle(image, (box[:2]), (box[2:4]), bbox_color, bbox_thickness)

    return image

def extract_bboxes(u_coords, v_coords) -> list:
    """ Extract bounding boxes from pixel coordinates

    Returns:
        tl_x, tl_y, br_x, br_y
    """
    min_u, max_u = min(u_coords), max(u_coords)
    min_v, max_v = min(v_coords), max(v_coords)

    return [min_u, min_v, max_u, max_v]


def load_transformations(yaml_path: str) -> dict:
    """ Load transformations from yaml file.

    The yaml file contains a transformation for each image taken for a specific scene.
    The format is as follows:
    -   image_id: 0
        origin_frame: /map
        rotation:
            w: 0.45350259463085363
            x: -0.8848484052531372
            y: 0.09218946856836649
            z: -0.053663751910806724
        target_frame: /head_rgbd_sensor_rgb_frame
        timestamp: 1567880058.4000113
        translation:
            x: -0.18408279805772904
            y: 1.0671824290226986
            z: 1.0897595086114673
    -   image_id: 1
        ...

    The function returns a dict with rotation, translation and timestamp for each entry.
    """
    with open(yaml_path, "r", encoding="utf-8") as stream:
        try:
            data = yaml.safe_load(stream)
            transformations = {}
            for entry in data:
                id = entry["id"]
                rotation = entry["rotation"]
                translation = entry["translation"]
                timestamp = entry["timestamp"]
                file_name = entry["file_name"]
                transformations[id] = {
                    "rotation": rotation, 
                    "translation": translation,
                    "timestamp": timestamp,
                    "file_name": file_name}
            return transformations
        except yaml.YAMLError as exc:
            print(exc)

def squeeze_img(image: np.array, intrinsics: Intrinsic, strength: float = 0.0005) -> np.array:
    """ Squeeze image with non-linear transformation.

    Args:
        image: image to squeeze
        intrinsics: object of instance Intrinsic
        strength: strength of the squeezing effect

    Returns:
        squeezed image
    """
    assert isinstance(intrinsics, Intrinsic), f"intrinsics must be instance of Intrinsic Class,\
                                                but is {type(intrinsics)}"
    assert isinstance(strength, float), f"strength must be float, but is {type(strength)}"

    # Create the maps for remapping
    map_x = np.zeros((intrinsics.height, intrinsics.width), dtype=np.float32)
    map_y = np.zeros((intrinsics.height, intrinsics.width), dtype=np.float32)

    # Define the center of the image
    center_x = intrinsics.width / 2

    for j in range(intrinsics.height):
        for i in range(intrinsics.width):
            # Offset from the center
            offset = (i - center_x)
            
            # Calculate the stretched position with a non-linear transformation
            stretched_position = center_x + offset * (1 + strength * abs(offset))
            
            # Ensure that the stretched position does not go out of the image bounds
            stretched_position = max(0, min(intrinsics.width - 1, stretched_position))
            
            # Assign the stretched position to the map
            map_x[j, i] = stretched_position
            map_y[j, i] = j

    # Remap the image
    squeezed_img = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    return squeezed_img

def squeeze_coordinates(coords: tuple, intrinsics: Intrinsic, strength: float = 0.0005) -> np.array:
    """ Squeeze image coordinates with non-linear transformation.

    Args:
        u_coords: u coordinates
        strength: strength of the squeezing effect

    Returns:
        squeezed u coordinates as integers
    """
    u_coords, v_coords = coords
    assert isinstance(u_coords, np.ndarray), f"u_coords must be numpy array, but is {type(u_coords)}"
    assert isinstance(strength, float), f"strength must be float, but is {type(strength)}"

      # Create the maps for remapping
    map_x = np.zeros((intrinsics.height, intrinsics.width), dtype=np.float32)
    map_y = np.zeros((intrinsics.height, intrinsics.width), dtype=np.float32)
    
    # Define the center of the image
    center_x = intrinsics.width / 2

    # Offset from the center
    offset = u_coords - center_x

    # Calculate the stretched position with a non-linear transformation
    stretched_position = center_x + offset * (1 - strength * abs(offset))
    stretched_position = np.clip(stretched_position, 0, intrinsics.width - 1)

    return stretched_position.astype(int), v_coords


if __name__ == "__main__":
    PCD_NAME: str = "../../data/annotation/office/scene4/merged_plane_clouds_ds002.pcd"
    # PCD_NAME: str = "./testmodel/model1.pcd"
    ANNO_NAME: str = "../../data/annotation/office/scene4/merged_plane_clouds_ds002_GT.anno"

    pcd = o3d.io.read_point_cloud(PCD_NAME)
    print(pcd)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    pcd, annotation_dict = annotate_pcd(pcd, ANNO_NAME)

    o3d.visualization.draw_geometries([pcd, mesh_frame])
