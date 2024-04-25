import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# print(torch.load("data/GH30_Office/all_target_bboxes.pt"))
# print(torch.load("data/GH30_Office/predictions/batch_image2_predicted_bboxes.pt"))
