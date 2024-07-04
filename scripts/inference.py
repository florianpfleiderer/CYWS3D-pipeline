#! usr/bin/env python3.9#
# Credits to Ragav Sachdeva
# MIT License
"""
Contains the Code for running an inference with cyws3d.
"""
import os
from tracemalloc import start
import time
import yaml
import logging
from importlib.metadata import version
from easydict import EasyDict
try:
    from src.inference.model import Model
    from src.inference.utils import create_batch_from_metadata, fill_in_the_missing_information, \
        prepare_batch_for_model, visualise_predictions, plot_correspondences, \
            undo_imagenet_normalization
    from src.inference.correspondence_extractor import CorrespondenceExtractor
except ImportError:
    from model import Model
    from utils import create_batch_from_metadata, fill_in_the_missing_information, \
        prepare_batch_for_model, visualise_predictions, plot_correspondences, \
            undo_imagenet_normalization
    from correspondence_extractor import CorrespondenceExtractor
import torch
try:
    from src.inference.geometry import remove_bboxes_with_area_less_than, \
        suppress_overlapping_bboxes, keep_matching_bboxes, filter_low_confidence_bboxes
except ImportError:
    from geometry import remove_bboxes_with_area_less_than, suppress_overlapping_bboxes, \
        keep_matching_bboxes, filter_low_confidence_bboxes
from src.globals import BBOX_AREA, CONFIDENCE_THRESHOLD, MAX_PREDICTIONS

# check required version of cyws3d-pipeline (defined in setup.py)
required_version = '1.0'
if version('cyws3d-pipeline') < required_version:
    raise ImportError(f"cyws3d-pipeline must be version {required_version}")

logging.basicConfig()
logger = logging.getLogger(__name__)

def main(
    config_file: str = "config.yml",
    # input_metadata: str = "data/inference/demo_data/input_metadata.yml",
    room : str = None,
    load_weights_from: str = "./cyws-3d.ckpt",
    filter_predictions_with_area_under: int = BBOX_AREA,
    keep_matching_bboxes_only: bool = False,
    max_predictions_to_display: int = MAX_PREDICTIONS,
    minimum_confidence_threshold: float = CONFIDENCE_THRESHOLD,
    log_level: str = "INFO"
):
    """ 
    runs the inference with cyws3d.
    """
    if room is None:
        raise ValueError("Please provide the room name as command line argument")
    input_metadata = f"data/GH30_{room}/input_metadata.yaml"

    save_path = os.path.join(
        input_metadata.split("/")[0], input_metadata.split("/")[1], "predictions")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    logger.setLevel(level=getattr(logging, log_level.upper()))
    # add a filehandler
    # file_handler = logging.FileHandler(f'{save_path}/logfile.log', 'w')
    # file_handler.setLevel(getattr(logging, log_level.upper()))
    # logger.addHandler(file_handler)
    logger.info("logger set to %s", logger.level)
    logger.info("Folder: %s\nParameters: %s, %s, %s, %s, %s, %s, %s, %s",
                save_path, config_file, input_metadata, room, load_weights_from,
                filter_predictions_with_area_under, keep_matching_bboxes_only,
                max_predictions_to_display, minimum_confidence_threshold)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    configs = get_easy_dict_from_yaml_file(config_file)
    model = Model(configs, load_weights_from=load_weights_from).to(device)
    correspondence_extractor = CorrespondenceExtractor(device=device)
    depth_predictor = torch.hub.load(
        "isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).eval().to(device)

    image1_predictions = []
    image2_predictions = []

    batch_metadata = get_easy_dict_from_yaml_file(input_metadata)
    full_batch = create_batch_from_metadata(batch_metadata, "cpu")
    batch_size = configs.batch_size
    if len(full_batch["image1"]) % batch_size != 0:
        number_of_batches = len(full_batch["image1"]) // batch_size + 1
    else:
        number_of_batches = len(full_batch["image1"]) // batch_size

    logger.info("Batch size: %s", batch_size)
    logger.info("Number of batches: %s", number_of_batches)
    img_cntr = 0
    for n in range(number_of_batches):
        torch.cuda.empty_cache()
        try:
            logger.debug(torch.cuda.memory_summary())
        except KeyError: # if the GPU is not available
            logger.debug("GPU not available")
        logger.info("Processing batch %s", n)

        batch = {key: value[n*batch_size:(n+1)*batch_size]\
             for key, value in full_batch.items()}
        for key in batch.keys():
            batch[key] = [item.to(device) if isinstance(item, torch.Tensor) else item \
                for item in batch[key]]

        start_time = time.time()
        batch = fill_in_the_missing_information(
            batch, depth_predictor, correspondence_extractor, device=device)
        logger.info("Time taken to fill in the missing information: %.2f seconds", \
            time.time() - start_time)
        start_time = time.time()
        batch = prepare_batch_for_model(batch, device=device)
        logger.info("Time taken to prepare the batch for the model: %.2f seconds", \
            time.time() - start_time)
        start_time = time.time()
        batch_image1_predicted_bboxes, batch_image2_predicted_bboxes = model.predict(batch)
        logger.info("Time taken to predict: %.2f seconds", time.time() - start_time)

        for i, (image1_bboxes, image2_bboxes) in enumerate(zip(batch_image1_predicted_bboxes,
                                                                batch_image2_predicted_bboxes)):
            logger.info("Processing image pair %s", img_cntr)
            # plot_correspondences(batch["image1"][i].cpu(), batch["image2"][i].cpu(),
            #                     batch["points1"][i].cpu(), batch["points2"][i].cpu(),
            #                     save_path=f"{save_path}/correspondences_{img_cntr}.png")
            image1_bboxes, image2_bboxes = \
                image1_bboxes[0].cpu().numpy(), image2_bboxes[0].cpu().numpy()
            image1_bboxes = remove_bboxes_with_area_less_than(
                image1_bboxes, filter_predictions_with_area_under)
            image2_bboxes = remove_bboxes_with_area_less_than(
                image2_bboxes, filter_predictions_with_area_under)
            logger.debug("suppressing overlapping bboxes for image pair %s", img_cntr)
            image1_bboxes, scores1 = \
                suppress_overlapping_bboxes(image1_bboxes[:, :4], image1_bboxes[:, 4])
            image2_bboxes, scores2 = \
                suppress_overlapping_bboxes(image2_bboxes[:, :4], image2_bboxes[:, 4])
            logger.debug("img02 bboxes after suppressing overlapping boxes: %s", image2_bboxes)
            logger.debug("img02 scores: %s", scores2)
            if keep_matching_bboxes_only:
                image1_bboxes, image2_bboxes = keep_matching_bboxes(
                    batch,
                    i,
                    image1_bboxes,
                    image2_bboxes,
                    scores1,
                    scores2,
                    minimum_confidence_threshold,
                    device=device
                )
            logger.debug("img02 bboxes after keep matching boxes: %s", image2_bboxes)
            logger.debug("img02 scores: %s", scores2)
            image1_bboxes, scores1 = filter_low_confidence_bboxes(
                image1_bboxes, scores1, minimum_confidence_threshold)
            image2_bboxes, scores2 = filter_low_confidence_bboxes(
                image2_bboxes, scores2, minimum_confidence_threshold)
            visualise_predictions(undo_imagenet_normalization(batch["image1"][i].cpu()),
                                    undo_imagenet_normalization(batch["image2"][i].cpu()),
                                    image1_bboxes[:max_predictions_to_display],
                                    image2_bboxes[:max_predictions_to_display],
                                        scores1[:max_predictions_to_display],
                                        scores2[:max_predictions_to_display],
                                            save_path=f"{save_path}/prediction_{img_cntr}.png")

            image1_predictions.append(dict(
                image=f"prediction_{img_cntr}", 
                boxes=torch.round(torch.as_tensor(
                    image1_bboxes[:max_predictions_to_display], dtype=torch.float32)),
                scores=torch.as_tensor(scores1[:max_predictions_to_display], dtype=torch.float32),
                labels=torch.zeros(
                    len(image1_bboxes[:max_predictions_to_display]), dtype=torch.int32))
                )
            image2_predictions.append(dict(
                image=f"prediction_{img_cntr}", 
                boxes=torch.round(torch.as_tensor(
                    image2_bboxes[:max_predictions_to_display], dtype=torch.float32)),
                scores=torch.as_tensor(scores2[:max_predictions_to_display], dtype=torch.float32),
                labels=torch.zeros(
                    len(image2_bboxes[:max_predictions_to_display]), dtype=torch.int32))
                )
            img_cntr += 1

    # save the batches for calculating mAP
    torch.save(image1_predictions, f'{save_path}/batch_image1_predicted_bboxes.pt')
    torch.save(image2_predictions, f'{save_path}/batch_image2_predicted_bboxes.pt')
    # Save configuration parameters to a YAML file
    configurations = {
        "filter_predictions_with_area_under": filter_predictions_with_area_under,
        "keep_matching_bboxes_only": keep_matching_bboxes_only,
        "max_predictions_to_display": max_predictions_to_display,
        "minimum_confidence_threshold": minimum_confidence_threshold
    }
    existing_configurations = {}
    existing_file_path = os.path.join(save_path, "metadata_configurations.yaml")
    if os.path.exists(existing_file_path):
        with open(existing_file_path, "r") as file:
            existing_configurations = yaml.safe_load(file)
    existing_configurations.update(configurations)
    with open(existing_file_path, "w") as file:
        yaml.dump(existing_configurations, file)

def get_easy_dict_from_yaml_file(path_to_yaml_file):
    """
    Reads a yaml and returns it as an easy dict.
    """
    with open(path_to_yaml_file, "r", encoding="utf-8") as stream:
        yaml_file = yaml.safe_load(stream)
    return EasyDict(yaml_file)

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
