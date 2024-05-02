import torch
import numpy as np
from pprint import pprint
from pprint import pp

# position1 = np.load("data/inference/demo_data/position1.npy")
# position2 = np.load("data/inference/demo_data/position2.npy")
# rotation1 = np.load("data/inference/demo_data/rotation1.npy")
# rotation2 = np.load("data/inference/demo_data/rotation2.npy")
# intrinsics = np.load("data/inference/demo_data/intrinsics1.npy")
# print(f"positions: {position1}, {position2}")
# print(f"\nrotations: {rotation1}, {rotation2}")
# print(f"\nintrinsics: {intrinsics}")

# position1 = np.load("data/GH30_Office/scene1/hsrb_head_rgbd_sensor_rgb_image_raw-1567879848-463321678_position.npy")
# position2 = np.load("data/GH30_Office/scene2/hsrb_head_rgbd_sensor_rgb_image_raw-1567880062-488732569_position.npy")
# rotation1 = np.load("data/GH30_Office/scene1/hsrb_head_rgbd_sensor_rgb_image_raw-1567879848-463321678_rotation.npy")
# rotation2 = np.load("data/GH30_Office/scene2/hsrb_head_rgbd_sensor_rgb_image_raw-1567880062-488732569_rotation.npy")
# intrinsics = np.load("data/ObChange/camera_info.npy")
# print(f"positions: {position1}, {position2}")
# print(f"\nrotations: {rotation1}, {rotation2}")
# print(f"\nintrinsics: {intrinsics}")

print("target bboxes:\n")
target_bboxes = torch.load("data/GH30_Office/all_target_bboxes.pt")
pprint(target_bboxes)
print("\n\npredictions:\n")
predicted_bboxes = torch.load("data/GH30_Office/predictions/batch_image2_predicted_bboxes.pt")
pprint(predicted_bboxes)
