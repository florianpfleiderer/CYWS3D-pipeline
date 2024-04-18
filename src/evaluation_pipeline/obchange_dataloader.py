# Created on Thu Apr 18 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
this module provides a class and functions for loading the obchange dataset with
naming scheme as follows:

TODO: add naming scheme like in documents folder
"""

import os
import json
import xml.etree.ElementTree as ET
import torch

class ObchangeDataset():
    ''' This class represents the rgb frames extracted from the obChange original input data
    '''
    def __init__(self, root_dir, transform=None):
        ''' Constructor for the ObchangeDataset class

        Args:
            root_dir (str): the directory containing the rgb frames
            transform (callable, optional): Optional transform to be applied on a sample
        '''
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        ''' Returns the length of the dataset
        '''
        return len(os.listdir(self.root_dir))
    
    def __getitem__(self, idx):
        ''' Returns the rgb frame at the given index
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, f'{idx}.png')
        image = io.imread(img_name)
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

def load_roboflow_export(json_path):
    ''' this function loads the json file exported from roboflow 
    (COCO format = x_min, y_min, width, height) and returns the data in the 
    correct format (x_min,y_min,x_max,y_max)

    Keep in mind that the resulting image should be 224x224 pixels
    '''
    targets = []

    with open(json_path) as file:
        gt = json.load(file)
        for i in range(len(gt['images'])):
            targets.append(dict(
                boxes = torch.tensor([[gt['annotations'][i]['bbox'][0], \
                    gt['annotations'][i]['bbox'][1], gt['annotations'][i]['bbox'][0] \
                        + gt['annotations'][i]['bbox'][2], gt['annotations'][i]['bbox'][1] \
                            + gt['annotations'][i]['bbox'][3]]], dtype=torch.float32),
                labels = torch.zeros(1, dtype=torch.int64)
            ))
    # rearrang elist to have the same order as the predictions
    targets = [targets[i] for i in [0, 2, 1, 3]]
    return targets

def load_pascal_voc_export(folder_path):
    ''' this function loads the xml file exported from pascal voc and returns the data in the 
    correct format (x_min,y_min,x_max,y_max)

    Keep in mind that the resulting image should be 224x224 pixels

    the folder structure for the files should be as follows:
    folder_path
    ├── img01<name>.jpg
    ├── img01<name>.xml
    ├── img02<name>.jpg
    ├── omg02<name>.xml
    ...

    where the xml file contains the infos about the image and bbox coordinates
    '''
    targets = []
    files = os.listdir(folder_path)
    files.sort()
    for file in files:
        if file.endswith('.xml'):
            tree = ET.parse(os.path.join(folder_path, file))
            root = tree.getroot()
            bboxes = []
            labels = []
            for obj in root.findall('object'):
                bbox = obj.find('bndbox')
                bboxes.append([int(bbox.find('xmin').text), int(bbox.find('ymin').text), \
                    int(bbox.find('xmax').text), int(bbox.find('ymax').text)])
                labels.append(0)
            targets.append(dict(
                boxes = torch.tensor(bboxes, dtype=torch.float32),
                labels = torch.tensor(labels, dtype=torch.int64)
            ))
    return targets