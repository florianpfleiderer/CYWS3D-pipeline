# Created on Thu Apr 18 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
This script contains classes to store intrinsic and extrinsic parameters of a camera and a
function to project a pointcloud onto 2d coordinates given the intrinsics and extrinsics according
to the classes defined below.
"""

import logging
import xml.etree.ElementTree as ET
import json
import yaml
import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

class Intrinsic():
    ''' Class to store intrinsic parameters of a camera

    Args:
        cx: principal point x
        cy: principal point y
        f: focal length
    '''
    def __init__(self, cx=None, cy=None, f=None, width=None, height=None):
        self.cx = cx
        self.cy = cy
        self.fx = f
        self.fy = f
        self.width = width
        self.height = height
        self.distortion = None

    def from_xml(self, xml_file: str):
        ''' Load the intrinsic parameters from an xml file

        Args:
            xml_file: relative path to the xml file
        '''
        root = ET.parse(xml_file).getroot()
        calibration = root.find(".//calibration")
        self.cx = float(calibration.find('cx').text)
        self.cy = float(calibration.find('cy').text)
        self.fx = float(calibration.find('f').text)
        self.fy = float(calibration.find('f').text)
        self.width = int(root.find(".//resolution").attrib['width'])
        self.height = int(root.find(".//resolution").attrib['height'])

    def from_json(self, json_file: str):
        ''' Load the intrinsic parameters from a json file

        Args:
            json_file: relative path to the json file
        '''
        with open(json_file, 'r', encoding='utf-8') as f:
            temp = json.load(f)
            data = np.array(temp['K']).reshape(3, 3)
            self.cx = data[0][2]
            self.cy = data[1][2]
            self.fx = data[0][0]
            self.fy = data[1][1]
            self.height = temp['height']
            self.width = temp['width']
            self.distortion = temp['D']

    def matrix(self):
        ''' Return the intrinsic matrix of the camera '''
        return np.array([[self.fx, 0, self.cx],
                        [0, self.fy, self.cy],
                        [0, 0, 1]])

    def homogenous_matrix(self):
        ''' Return the homogenous intrinsic matrix of the camera '''
        return np.array([[self.fx, 0, self.cx, 0],
                        [0, self.fy, self.cy, 0],
                        [0, 0, 1, 0]])

    def distortion_coeffs(self) -> np.array:
        """ return distortion as np.array
        """
        return np.asarray(self.distortion)


class Extrinsic():
    ''' Class to store extrinsic parameters of a camera

    Args:
        M: 4x4 homogenous extrinsic matrix
        position: position of the camera (in world coordinates)
        rotation: rotation matrix of the camera (in world coordinates)
    '''
    def __init__(self, position=None, rotation=None):
        self.extrinsic_matrix = np.eye(4)
        self.position = position
        self.rotation = rotation

    def from_xml(self, xml_file: str):
        ''' Load the viewpoint from an xml file and calculate the extrinsic matrix

        Args:
            xml_file: relative path to the xml file
        '''
        root = ET.parse(xml_file).getroot()
        transform = root.find('.//camera[@id="1"]/transform').text
        self.extrinsic_matrix = np.asarray([transform.split(' ')]).astype(np.float32).reshape(4, 4)

    def from_json(self, json_file: str, scale: int = 1):
        ''' Load the viewpoint from a json file and calculate the extrinsic matrix

        Args:
            json_file: relative path to the json file
        '''
        with open(json_file, 'r', encoding='utf-8') as f:
            trajectory = json.load(f)['trajectory'][0]

        lookat = np.asarray(trajectory['lookat'])
        front = np.asarray(trajectory['front'])
        up = np.asarray(trajectory['up'])
        zoom = trajectory['zoom']

        self.position = lookat + front * zoom * scale

        # Normalize the vectors
        def normalize(v):
            norm = np.linalg.norm(v)
            if norm == 0:
                raise ValueError("Cannot normalize a vector with zero magnitude")
            return v / norm

        front = normalize(front)
        up = normalize(up)
        right = normalize(np.cross(up, front))
        up = np.cross(front, right)  # 'up' is recalculated to ensure orthogonality

        # Construct the rotation matrix
        self.rotation = np.column_stack((right, up, front))

        # Construct the rotation matrix
        self.rotation = np.column_stack((right, up, front))

        self.extrinsic_matrix[:3, :4] = np.column_stack((self.rotation, self.position))
        self.extrinsic_matrix = np.linalg.inv(self.extrinsic_matrix)
    
    def from_yaml(self, yaml_file: str):
        ''' load the transformation from the yaml file containing the transformation from 
        map origin to camera_center_frame in tf tree at a specific timestamp.

        File format:
        timestamp:
        transformation:
            rotation:
                x: 0.0
                y: 0.0
                z: 0.0
                w: 1.0
            translation:
                x: 0.0
                y: 0.0
                z: 0.0
        '''
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            rotation = data['transformation']['rotation']
            self.rotation = self.quat_to_rot(rotation)
            translation = data['transformation']['translation']
            self.position = np.array([translation['x'], translation['y'], translation['z']])

        self.extrinsic_matrix[:3, :4] = np.column_stack((self.rotation, self.position))
        self.extrinsic_matrix = np.linalg.inv(self.extrinsic_matrix)

    def from_dict(self, data: dict):
        """load the transformation from a dictionary containing the transformation from 
        map origin to camera_center_frame in tf tree at a specific timestamp.

        dict format:
        {'id': 0, 
        'origin_frame': '/map', 
        'rotation': {
            'w': 0.45350259463085363, 
            'x': -0.8848484052531372, 
            'y': 0.09218946856836649, 
            'z': -0.053663751910806724}, 
        'target_frame': '/head_rgbd_sensor_rgb_frame', 
        'timestamp': 1567880058.4000113, 
        'translation': {
            'x': -0.18408279805772904, 
            'y': 1.0671824290226986, 
            'z': 1.0897595086114673}
        }
        """
        rotation = data['rotation']
        self.rotation = self.quat_to_rot(rotation)
        translation = data['translation']
        self.position = np.array([translation['x'], translation['y'], translation['z']])

        self.extrinsic_matrix[:3, :4] = np.column_stack((self.rotation, self.position))
        self.extrinsic_matrix = np.linalg.inv(self.extrinsic_matrix)
    
    def quat_to_rot(self, Q):
        '''
        Covert a quaternion into a full three-dimensional rotation matrix.
    
        Input
        :param Q: a dictionary with four keys: x, y, z, w 
    
        Output
        :return: A 3x3 element matrix representing the full 3D rotation matrix. 
                This rotation matrix converts a point in the local reference 
                frame to a point in the global reference frame.
        '''
        # Extract the values from Q
        q0 = Q['w']
        q1 = Q['x']
        q2 = Q['y']
        q3 = Q['z']
        
        # First row of the rotation matrix
        r00 = 2 * (q0 * q0 + q1 * q1) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 * q0 + q2 * q2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 * q0 + q3 * q3) - 1
        
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])        
        return rot_matrix

    def from_yaml(self, yaml_file: str):
        ''' load the transformation from the yaml file containing the transformation from 
        map origin to camera_center_frame in tf tree at a specific timestamp.

        File format:
        timestamp:
        transformation:
            rotation:
                x: 0.0
                y: 0.0
                z: 0.0
                w: 1.0
            translation:
                x: 0.0
                y: 0.0
                z: 0.0
        '''
        with open(yaml_file, 'r', encoding="utf-8") as f:
            data = yaml.safe_load(f)
            rotation = data['transformation']['rotation']
            self.rotation = self.quat_to_rot(rotation)
            translation = data['transformation']['translation']
            self.position = np.array([translation['x'], translation['y'], translation['z']])

        self.extrinsic_matrix[:3, :4] = np.column_stack((self.rotation, self.position))
        self.extrinsic_matrix = np.linalg.inv(self.extrinsic_matrix)

    def from_dict(self, data: dict):
        """load the transformation from a dictionary containing the transformation from 
        map origin to camera_center_frame in tf tree at a specific timestamp.

        dict format:
        {'id': 0, 
        'origin_frame': '/map', 
        'rotation': {
            'w': 0.45350259463085363, 
            'x': -0.8848484052531372, 
            'y': 0.09218946856836649, 
            'z': -0.053663751910806724}, 
        'target_frame': '/head_rgbd_sensor_rgb_frame', 
        'timestamp': 1567880058.4000113, 
        'translation': {
            'x': -0.18408279805772904, 
            'y': 1.0671824290226986, 
            'z': 1.0897595086114673}
        }
        """
        rotation = data['rotation']
        self.rotation = self.quat_to_rot(rotation)
        translation = data['translation']
        self.position = np.array([translation['x'], translation['y'], translation['z']])

        self.extrinsic_matrix[:3, :4] = np.column_stack((self.rotation, self.position))
        self.extrinsic_matrix = np.linalg.inv(self.extrinsic_matrix)

    def matrix(self):
        ''' Return the extrinsic matrix of the camera '''
        return self.extrinsic_matrix[:3, :4]

    def homogenous_matrix(self):
        ''' Return the homogenous extrinsic matrix of the camera '''
        return self.extrinsic_matrix


def inside_frustum(point, fov_x, fov_y, near, far) -> bool:
    ''' Check if point is inside frustum '''
    x, y, z = point
    if z > near or z < far:
        return False
    if abs(x) > abs(z * np.tan(np.radians(fov_x / 2))):
        return False
    if abs(y) > abs(z * np.tan(np.radians(fov_y / 2))):
        return False
    return True


def frustum_culling(points: np.array, fov_x: float, fov_y: float) -> list:
    ''' Filter points based on frustum culling 
    
    Args:
        points: np.array of shape (N, 3)
        fov: field of view
        
    Returns:
        list of filtered points indices
    '''

    far = min(points[:, 2])
    near = max(points[:, 2])

    indices = [i for i, point in enumerate(points) if inside_frustum(point, fov_x, fov_y, near, far)]
    if len(indices) == 0:
        raise ValueError("No points in frustum")
    return indices


def project_to_2d(points_pos: np.array, K: np.array, D: np.array, width: int = 640, height: int = 480) \
                  -> (np.array, np.array):
    ''' use pinhole model to project a pointcloud onto 2d coordinates:

    Args:
        points_pos: Nx3 array of 3d points in camera coordinate system
        K: 3x4 homogenous intrinsic matrix
            | fx 0  cx 0 |
            | 0  fy cy 0 |
            | 0  0  1  0 |
        D: 1x5 distortion coefficients
        width: width of the image in pixels
        height: height of the image in pixels
    
    Returns:
        u: x coordinates in image plane
        v: y coordinates in image plane
    '''
    assert points_pos.shape[1] == 3, "points_pos must be Nx3"
    assert K.shape == (3, 4), "K must be 3x4"
    assert D.shape == (5,), "D must be 1x5"

    # homogenous coordinates
    hom_coord = np.hstack((points_pos, np.ones((points_pos.shape[0], 1))))

    # projection
    proj = K @ hom_coord.T
    logger.debug(proj[:2, 40:50])

    proj[:2, :] = apply_distortion(proj[:2, :].T, K, D).T
    logger.debug(proj[:2, 40:50])

    u = np.round(proj[0, :] / proj[2, :]).astype(int).clip(0, width-1)
    v = np.round(proj[1, :] / proj[2, :]).astype(int).clip(0, height-1)

    return u, v

def apply_distortion(points_2d, K, D):
    """
    Applies the radial and tangential distortion to 2D points using the plumb bob model.

    Args:
        points_2d: Nx2 array of undistorted 2D points.
        K: 3x4 camera intrinsic matrix.
        D: 1x5 array of distortion coefficients (k1, k2, p1, p2, k3).

    Returns:
        distorted_points: Nx2 array of distorted 2D points.
    """
    # Extract the focal lengths and principal point from K
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Normalize the coordinates
    x = (points_2d[:, 0] - cx) / fx
    y = (points_2d[:, 1] - cy) / fy

    # Compute the radial distortion
    r2 = x**2 + y**2
    radial_distortion = 1 + D[0] * r2 + D[1] * r2**2 + D[4] * r2**3

    # Compute the tangential distortion
    xy = x * y
    x_distorted = x * radial_distortion + 2 * D[2] * xy + D[3] * (r2 + 2 * x**2)
    y_distorted = y * radial_distortion + D[2] * (r2 + 2 * y**2) + 2 * D[3] * xy

    # Scale back to image coordinates
    x_final = x_distorted * fx + cx
    y_final = y_distorted * fy + cy

    return np.vstack((x_final, y_final)).T


if __name__ == "__main__":
    # load pcd file
    pcd_name: str = "./testmodel/model1.pcd"
    pcd = o3d.io.read_point_cloud(pcd_name)

    # Build intrinsic matrix
    intrinsics = Intrinsic()
    intrinsics.from_xml("./testmodel/model1_cameras.xml")

    K = intrinsics.homogenous_matrix()
    logger.debug(K)

    # extrinsic matrix
    extrinsics = Extrinsic()
    extrinsics.from_json("./testmodel/model1_viewpoint_00.json", scale = 10)
    M = extrinsics.homogenous_matrix()

    logger.debug(M)
