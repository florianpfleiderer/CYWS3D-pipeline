import torch
import numpy as np

positions = np.load("data/inference/demo_data/position1.npy")
rotations = np.load("data/inference/demo_data/rotation1.npy")
intrinsics = np.load("data/inference/demo_data/intrinsics1.npy")
print(f"positions: {positions}")
print(f"\nrotations: {rotations}")
print(f"\nintrinsics: {intrinsics}")


# print("target bboxes:\n")
# print(torch.load("data/GH30_Office/all_target_bboxes.pt"))
# print("\n\npredictions:\n")
# print(torch.load("data/GH30_Office/predictions/batch_image2_predicted_bboxes.pt"))
