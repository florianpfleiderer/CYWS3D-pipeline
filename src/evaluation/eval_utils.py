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
from pprint import pprint

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
        mar_10=mAP['mar_10'],
        mar_100=mAP['mar_100'],
        mar_small=mAP['mar_small'],
        mar_medium=mAP['mar_medium'],
        mar_large=mAP['mar_large'],
        ious=dict(),
        precision=mAP['precision'],
        recall=mAP['recall'],
        scores=mAP['scores']
    )
    for key, value in mAP['ious'].items():
        if isinstance(value, torch.Tensor):
            mAP['ious'][key] = value.squeeze().cpu().tolist()
            logger.debug(f"converted {key} from tensor to list")
        mAP_json['ious'][int(key[0])] = mAP['ious'][key]

    with open(filepath, 'w') as f:
        json.dump(mAP_json, f, indent=2)

def load_map_from_json(filepath) -> dict:
    """ This functions loads teh file saved by save_map_as_json() and converts it to 
    original format, hvaing tensors as values.
    """
    with open(filepath, 'r') as f:
        mAP_json = json.load(f)

    mAP = dict(
        map=torch.tensor(mAP_json['map']),
        map_50=torch.tensor(mAP_json['map_50']),
        map_75=torch.tensor(mAP_json['map_75']),
        map_small=torch.tensor(mAP_json['map_small']),
        map_medium=torch.tensor(mAP_json['map_medium']),
        map_large=torch.tensor(mAP_json['map_large']),
        mar_1=torch.tensor(mAP_json['mar_1']),
        mar_10=torch.tensor(mAP_json['mar_10']),
        mar_100=torch.tensor(mAP_json['mar_100']),
        mar_small=torch.tensor(mAP_json['mar_small']),
        mar_medium=torch.tensor(mAP_json['mar_medium']),
        mar_large=torch.tensor(mAP_json['mar_large']),
        ious=dict(),
        precision=torch.tensor(mAP_json['precision']),
        recall=torch.tensor(mAP_json['recall']),
        scores=torch.tensor(mAP_json['scores'])
    )
    for key, value in mAP_json['ious'].items():
        mAP['ious'][(key, 0)] = torch.tensor(value)
    return mAP

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
    # pprint(f"image2_set: {image2_set}")
    filtered_boxes = [entry for entry in target_bboxes if entry['image'] in image2_set]
    # pprint(f"filtered {len(target_bboxes)-len(filtered_boxes)} target bboxes")
    image2_list = [entry['image2'] for entry in targets_metadata]
    sorted_boxes = sorted(filtered_boxes, key=lambda x: image2_list.index(x['image']))
    # pprint(sorted_boxes)
    return sorted_boxes