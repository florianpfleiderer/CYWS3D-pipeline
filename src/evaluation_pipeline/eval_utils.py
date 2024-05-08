# Created on Wed May 08 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien
"""
Insert Module Description Here
"""
import logging
import json
import yaml
import torch
import numpy as np

logger = logging.getLogger(__name__)

def map_to_numpy(mAP) -> None:
    ''' this function converts the mAP dict to numpy arrays
    '''
    for key, value in mAP.items():
        if isinstance(value, torch.Tensor):
            mAP[key] = value.squeeze().cpu().numpy()

def map_to_list(mAP) -> None:
    ''' this function converts the mAP dict to lists
    '''
    for key, value in mAP.items():
        if isinstance(value, torch.Tensor):
            mAP[key] = value.squeeze().cpu().tolist()
            logger.debug(f"converted {key} from tensor to list")
        if isinstance(value, np.ndarray):
            mAP[key] = value.tolist()
            logger.debug(f"converted {key} from numpy array to list")

def save_map_as_json(mAP, filepath) -> None:
    ''' this function saves the mAP dict to a json file
    '''
    map_to_list(mAP)
    mAP_json = dict(
        map=mAP['map'],
        map_50=mAP['map_50'],
        map_75=mAP['map_75'],
        map_small=mAP['map_small'],
        map_medium=mAP['map_medium'],
        map_large=mAP['map_large'],
        mar_1=mAP['mar_1'],
        mar_3=mAP['mar_3'],
        mar_5=mAP['mar_5'],
        mar_small=mAP['mar_small'],
        mar_medium=mAP['mar_medium'],
        mar_large=mAP['mar_large']
    )
    with open(filepath, 'w') as f:
        json.dump(mAP_json, f)

def prepare_target_bboxes(target_bboxes, metadata_path) -> list:
    """ 
    This function prepares the target bboxes for evaluation. Only boxes which are used in
    infernce are filtered from all_traget_bboxes.

    Args:
        target_bboxes (list): list of dictionaries containing the target bboxes
        metadata_path (str): path to the metadata file used for inference
    
    Returns:
        list: list of dictionaries containing the target bboxes sorted according to the 
            metadata used for inference
    """
    with open(metadata_path, 'r') as f:
        targets_metadata = yaml.safe_load(f)['batch']

    image2_set = {entry['image2'] for entry in targets_metadata}
    filtered_boxes = [entry for entry in target_bboxes if entry['image'] in image2_set]
    image2_list = [entry['image2'] for entry in targets_metadata]
    sorted_boxes = sorted(filtered_boxes, key=lambda x: image2_list.index(x['image']))
    return sorted_boxes