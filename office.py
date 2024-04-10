''' module for office dataset 

    - load point cloud
    - load annotation
    - load camera info
    - load viewpoint info
    - project to 2d
    - draw 2d bboxes
    - show image
'''
import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
from annotation_pipeline.projection import Intrinsic, Extrinsic, project_to_2d, frustum_culling
import annotation_pipeline.utils as utils


FOLDER = "data/annotation/office/"
SCENE = "scene4/"
PCD_PATH = "merged_plane_clouds_ds002.pcd"
ANNO_PATH = "merged_plane_clouds_ds002_GT.anno"
CAMERA_INFO_JSON_PATH = "camera_info.json"
VIEWPOINT_INFO_JSON_PATH = "viewpoint_info.json"
gt_colour = np.array([0.1, 0.90, 0.1])

# load pcd file
pcd = o3d.io.read_point_cloud("./" + FOLDER + SCENE + PCD_PATH)

# ground truth
_, anno_dict = utils.annotate_pcd(pcd, "./" + FOLDER + SCENE + ANNO_PATH, gt_colour)
# bboxes = utils.extract_3d_bboxes(pcd, anno_dict, result=False)
# o3d.visualization.draw_geometries([pcd, bboxes[0], bboxes[1], bboxes[2]])

# intrinsic matrix
intrinsics = Intrinsic()
intrinsics.from_json("./"+FOLDER+CAMERA_INFO_JSON_PATH)

# extrinsic matrix
extrinsics = Extrinsic()
extrinsics.from_json("./"+FOLDER+VIEWPOINT_INFO_JSON_PATH)

# create mesh for showing the origin
mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

# create extrinsic matrix from 025_mug (1): 0.913546 -0.406737 0 1.66914 0.406737 0.913546 0 1.76407 0 0 1 0.74 0 0 0 1 
# M = np.array([[0.913546, -0.406737, 0, 1.66914], [0.406737, 0.913546, 0, 1.76407], [0, 0, 1, 0.74], [0, 0, 0, 1]])
# pcd.transform(M)
pcd.transform(extrinsics.homogenous_matrix())
points_pos = np.asarray(pcd.points)
points_color = np.asarray(pcd.colors)

# indices change after "select by index" (new pcd is created)
gt_pcd = pcd.select_by_index(anno_dict['077_rubiks_cube'])
gt_points_pos = np.asarray(gt_pcd.points)
gt_points_color = np.asarray(gt_pcd.colors)

# frustum culling
pcd = pcd.select_by_index(frustum_culling(points_pos, 60))
gt_pcd = gt_pcd.select_by_index(frustum_culling(gt_points_pos, 60))

points_pos = np.asarray(pcd.points)
points_color = np.asarray(pcd.colors)
gt_points_pos = np.asarray(gt_pcd.points)
gt_points_color = np.asarray(gt_pcd.colors)

# project and draw bboxes
u_coords, v_coords = project_to_2d(points_pos, \
                                    intrinsics.homogenous_matrix(), \
                                    480, \
                                    intrinsics.height)

gt_u, gt_v = project_to_2d(gt_points_pos, intrinsics.homogenous_matrix(), 480, intrinsics.height)

image = utils.draw_2d_bboxes((u_coords, v_coords), points_color, (gt_u, gt_v), intrinsics)

image = utils.draw_2d_bboxes_on_img("./data/annotation/office/scene4/img03_s4.png", gt_u, gt_v)

plt.imsave("image.png", image)
# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
