''' This script contains classes to store intrinsic and extrinsic parameters of a camera and a
function to project a pointcloud onto 2d coordinates given the intrinsics and extrinsics according
to the classes defined below.
'''
import xml.etree.ElementTree as ET
import json
import open3d as o3d
import numpy as np

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
        self.f = f
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
        self.f = float(calibration.find('f').text)
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
            self.f = data[0][0]
            self.distortion = np.array(temp['D'])
            self.height = temp['height']
            self.width = temp['width']


    def matrix(self):
        ''' Return the intrinsic matrix of the camera '''
        return np.array([[self.f, 0, self.cx],
                        [0, self.f, self.cy],
                        [0, 0, 1]])

    def homogenous_matrix(self):
        ''' Return the homogenous intrinsic matrix of the camera '''
        return np.array([[self.f, 0, self.cx, 0],
                        [0, self.f, self.cy, 0],
                        [0, 0, 1, 0]])


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

    def matrix(self):
        ''' Return the extrinsic matrix of the camera '''
        return self.extrinsic_matrix[:3, :4]

    def homogenous_matrix(self):
        ''' Return the homogenous extrinsic matrix of the camera '''
        return self.extrinsic_matrix


def inside_frustum(point, fov, near, far):
    ''' Check if point is inside frustum '''
    x, y, z = point
    if z > near or z < far:
        return False
    if abs(x) > abs(z * np.tan(np.radians(fov / 2))):
        return False
    if abs(y) > abs(z * np.tan(np.radians(fov / 2))):
        return False
    return True


def frustum_culling(points: np.array, fov: int) -> None:
    ''' Filter points based on frustum culling 
    
    Args:
        points: np.array of shape (N, 3)
        fov: field of view
        
    Returns:
        list of filtered points indices
    '''

    far = min(points[:, 2])
    near = max(points[:, 2])

    indices = [i for i, point in enumerate(points) if inside_frustum(point, fov, near, far)]
    return indices


def project_to_2d(points_pos: np.array, K: np.array, width: int = 640, height: int = 480) \
                  -> (np.array, np.array):
    ''' use pinhole model to project a pointcloud onto 2d coordinates:

    Args:
        points_pos: Nx3 array of 3d points in camera coordinate system
        K: 3x4 homogenous intrinsic matrix
            | fx 0  cx 0 |
            | 0  fy cy 0 |
            | 0  0  1  0 |
        width: width of the image in pixels
        height: height of the image in pixels
    
    Returns:
        u: x coordinates in image plane
        v: y coordinates in image plane
    '''
    assert points_pos.shape[1] == 3, "points_pos must be Nx3"
    assert K.shape == (3, 4), "K must be 3x4"

    # homogenous coordinates
    hom_coord = np.hstack((points_pos, np.ones((points_pos.shape[0], 1))))

    # projection
    proj = K @ hom_coord.T
    u = np.round(proj[0, :] / proj[2, :]).astype(int).clip(0, width-1)
    v = np.round(proj[1, :] / proj[2, :]).astype(int).clip(0, height-1)

    return u, v


if __name__ == "__main__":
    # load pcd file
    pcd_name: str = "./testmodel/model1.pcd"
    pcd = o3d.io.read_point_cloud(pcd_name)

    # Build intrinsic matrix
    intrinsics = Intrinsic()
    intrinsics.from_xml("./testmodel/model1_cameras.xml")

    K = intrinsics.homogenous_matrix()
    print(K)

    # extrinsic matrix
    extrinsics = Extrinsic()
    extrinsics.from_json("./testmodel/model1_viewpoint_00.json", scale = 10)
    M = extrinsics.homogenous_matrix()
    print(M)
