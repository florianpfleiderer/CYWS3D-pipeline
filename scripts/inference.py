# Credits to Ragav Sachdeva
# MIT License
"""
Contains the Code for running an inference with cyws3d.
"""
from tracemalloc import start
import time
import yaml
from easydict import EasyDict
try:
    from src.modules.model import Model
    from src.modules.utils import create_batch_from_metadata, fill_in_the_missing_information, \
        prepare_batch_for_model, visualise_predictions, plot_correspondences, \
            undo_imagenet_normalization
    from src.modules.correspondence_extractor import CorrespondenceExtractor
except ImportError:
    from model import Model
    from utils import create_batch_from_metadata, fill_in_the_missing_information, \
        prepare_batch_for_model, visualise_predictions, plot_correspondences, \
            undo_imagenet_normalization
    from correspondence_extractor import CorrespondenceExtractor
import torch
try:
    from src.modules.geometry import remove_bboxes_with_area_less_than, \
        suppress_overlapping_bboxes, keep_matching_bboxes, filter_low_confidence_bboxes
except ImportError:
    from geometry import remove_bboxes_with_area_less_than, suppress_overlapping_bboxes, \
        keep_matching_bboxes, filter_low_confidence_bboxes

def main(
    config_file: str = "config.yml",
    # input_metadata: str = "data/inference/demo_data/input_metadata.yml",
    input_metadata: str = "data/GH30_Office/input_metadata.yml",
    load_weights_from: str = "./cyws-3d.ckpt",
    filter_predictions_with_area_under: int = 400,
    keep_matching_bboxes_only: bool = True,
    max_predictions_to_display: int = 5,
    minimum_confidence_threshold: float = 0.1,
):
    """ 
    runs the inference with cyws3d.
    """

    configs = get_easy_dict_from_yaml_file(config_file)
    model = Model(configs, load_weights_from=load_weights_from)
    correspondence_extractor = CorrespondenceExtractor()
    depth_predictor = torch.hub.load("isl-org/ZoeDepth", "ZoeD_NK", pretrained=True).eval()

    batch_metadata = get_easy_dict_from_yaml_file(input_metadata)
    batch = create_batch_from_metadata(batch_metadata)
    start_time = time.time()
    batch = fill_in_the_missing_information(batch, depth_predictor, correspondence_extractor)
    print(f"Time taken to fill in the missing information: {time.time() - start_time:.2f} seconds")
    start_time = time.time()
    batch = prepare_batch_for_model(batch)
    print(f"Time taken to prepare the batch for the model: {time.time() - start_time:.2f} seconds")
    start_time = time.time()
    batch_image1_predicted_bboxes, batch_image2_predicted_bboxes = model.predict(batch)
    image1_predictions = []
    image2_predictions = []
    print(f"Time taken to predict: {time.time() - start_time:.2f} seconds")
    for i, (image1_bboxes, image2_bboxes) in enumerate(zip(batch_image1_predicted_bboxes,
                                                           batch_image2_predicted_bboxes)):
        print(f"Processing image pair {i}")
        # plot_correspondences(batch["image1"][i], batch["image2"][i], 
        #                      batch["points1"][i], batch["points2"][i], 
#                              save_path=f"predictions/correspondences_{i}.png")
        image1_bboxes, image2_bboxes = \
            image1_bboxes[0].cpu().numpy(), image2_bboxes[0].cpu().numpy()
        image1_bboxes = remove_bboxes_with_area_less_than(
            image1_bboxes, filter_predictions_with_area_under)
        image2_bboxes = remove_bboxes_with_area_less_than(
            image2_bboxes, filter_predictions_with_area_under)
        image1_bboxes, scores1 = \
            suppress_overlapping_bboxes(image1_bboxes[:, :4], image1_bboxes[:, 4])
        image2_bboxes, scores2 = \
            suppress_overlapping_bboxes(image2_bboxes[:, :4], image2_bboxes[:, 4])
        if keep_matching_bboxes_only:
            image1_bboxes, image2_bboxes = keep_matching_bboxes(
                batch,
                i,
                image1_bboxes,
                image2_bboxes,
                scores1,
                scores2,
                minimum_confidence_threshold,
            )
        image1_bboxes, scores1 = filter_low_confidence_bboxes(
            image1_bboxes, scores1, minimum_confidence_threshold)
        image2_bboxes, scores2 = filter_low_confidence_bboxes(
            image2_bboxes, scores2, minimum_confidence_threshold)

        visualise_predictions(undo_imagenet_normalization(batch["image1"][i]),
                              undo_imagenet_normalization(batch["image2"][i]),
                                image1_bboxes[:max_predictions_to_display],
                                image2_bboxes[:max_predictions_to_display],
                                    scores1[:max_predictions_to_display],
                                    scores2[:max_predictions_to_display],
                                        save_path=f"data/predictions/prediction_{i}.png")

        image1_predictions.append(dict(
            boxes=torch.as_tensor(image1_bboxes[:max_predictions_to_display], dtype=torch.float32),
            scores=torch.as_tensor(scores1[:max_predictions_to_display], dtype=torch.float32),
            labels=torch.zeros(len(image1_bboxes[:max_predictions_to_display]), dtype=torch.int64))
            )
        image2_predictions.append(dict(
            boxes=torch.as_tensor(image2_bboxes[:max_predictions_to_display], dtype=torch.float32),
            scores=torch.as_tensor(scores2[:max_predictions_to_display], dtype=torch.float32),
            labels=torch.zeros(len(image2_bboxes[:max_predictions_to_display]), dtype=torch.int64))
            )

    # save the batches for calculating mAP
    torch.save(image1_predictions, 'data/predictions/batch_image1_predicted_bboxes.pt')
    torch.save(image2_predictions, 'data/predictions/batch_image2_predicted_bboxes.pt')

def get_easy_dict_from_yaml_file(path_to_yaml_file):
    """
    Reads a yaml and returns it as an easy dict.
    """
    with open(path_to_yaml_file, "r") as stream:
        yaml_file = yaml.safe_load(stream)
    return EasyDict(yaml_file)

if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main)
