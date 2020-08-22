"""
python TestNetwork.py --config=standard --weights=../models/cellSegmentation/mask_rcnn_cell_0009.h5 --img_dir=../datasets/DatasetRGBChannels/train --save_dir=../results/Standard
python TestNetwork.py --config=standard --weights=../models/cellSegmentation_geneChannels/mask_rcnn_cell_0010.h5 --img_dir=../datasets/DatasetGeneChannels/val --save_dir=../results/Multi-channel/val
"""

import os
import sys
import re
import copy
import json
import time
import skimage.io
import skimage.draw
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append("../")
from mrcnn import model as modellib, utils, visualize as viz
from cell.CellMaskRCNN import \
    CellConfig as StandardConfig, \
    detect as single_detect
from cell.CellMultiModelDetection import \
    CellConfigNuclei as MultiNetworkNucleiConfig, \
    CellConfigGenes as MultiNetworkGenesConfig, \
    detect as multi_detect, bbox as extract_bbox, \
    compute_iou

############################################################
# Support functions
############################################################
GT_CLASSES = {
    "negative": 0,
    "red": 1,
    "white": 2,
    "red and white": 3
}
COLOUR_DICT = {
    0: (0.0, 0.5, 1.0),
    1: (1.0, 0.0, 0.0),
    2: (1.0, 1.0, 1.0),
    3: (1.0, 0.5, 0.0)
}


def load_ground_truth(dir, img):
    # Load the annotations
    filename = os.path.join(dir, "via_mask_annotations_json.json")
    annotations = json.load(open(filename))
    annotations = list(annotations.values())

    # Load the ground truths
    ground_truth = {}
    for annotation in annotations:
        # If only interested in one image, skip until it is found
        if img and annotation["filename"] != img:
            continue

        img_name = annotation["filename"]
        polygons = [r["shape_attributes"] for r in annotation["regions"]]
        classes = [r["region_attributes"] for r in annotation["regions"]]

        ground_truth[img_name] = (polygons, classes)

    return ground_truth


def evaluate_results(masks, boxes, classes, gt):
    res = {
        "average_iou": 0,
        "all_ious": [],
        "nuclei_true_positive": 0,
        "nuclei_false_positive": 0,
        "nuclei_false_negative": 0,
        "none_true_positive": 0,
        "none_false_positive": 0,
        "none_true_negative": 0,
        "none_false_negative": 0,
        "red_true_positive": 0,
        "red_false_positive": 0,
        "red_true_negative": 0,
        "red_false_negative": 0,
        "white_true_positive": 0,
        "white_false_positive": 0,
        "white_true_negative": 0,
        "white_false_negative": 0,
        "red_and_white_true_positive": 0,
        "red_and_white_false_positive": 0,
        "red_and_white_true_negative": 0,
        "red_and_white_false_negative": 0,
    }
    shape = masks[:, :, 0].shape
    
    # Check on nuclei detected
    gt_detected = []
    for i in range(len(classes)):
        m = masks[:, :, i]
        b = boxes[i]
        c = classes[i]

        # Find best match
        max_iou = 0
        gt_idx = None
        act_id = None
        for gt_i, (gt_pol, gt_class) in enumerate(zip(gt[0], gt[1])):
            # Extract ground truth values
            gt_class_id = GT_CLASSES[gt_class["gene expression"]]
            gt_mask = np.zeros(shape, dtype=np.uint8)
            rr, cc = skimage.draw.polygon(gt_pol["all_points_y"], gt_pol["all_points_x"])
            gt_mask[rr, cc] = 1
            gt_mask = gt_mask.astype(np.bool)
            gt_box = extract_bbox(gt_mask)

            iou = compute_iou(m, b, gt_mask, gt_box)
            if iou > max_iou:
                max_iou = iou
                act_id = gt_class_id
                gt_idx = gt_i

        # Compare best match
        if max_iou > 0.4:
            # Nucleus was present in the ground truth
            res["nuclei_true_positive"] += 1
            res["average_iou"] += max_iou / len(classes)
            res["all_ious"].append(max_iou)
            gt_detected.append(gt_idx)

            if act_id == c:
                # Correct classification
                if c == 0:
                    res["none_true_positive"] += 1
                    res["red_true_negative"] += 1
                    res["white_true_negative"] += 1
                    res["red_and_white_true_negative"] += 1
                elif c == 1:
                    res["red_true_positive"] += 1
                    res["white_true_negative"] += 1
                elif c == 2:
                    res["white_true_positive"] += 1
                    res["red_true_negative"] += 1
                elif c == 3:
                    res["red_and_white_true_positive"] += 1
                    res["none_true_negative"] += 1
            else:
                # Wrong classification
                if c == 0:
                    res["none_false_positive"] += 1
                elif c == 1:
                    res["red_false_positive"] += 1
                elif c == 2:
                    res["white_false_positive"] += 1
                elif c == 3:
                    res["red_and_white_false_positive"] += 1
        else:
            # Nucleus was not present in the ground truth
            res["nuclei_false_positive"] += 1

            # Classification must be wrong -> False positive
            if c == 0:
                res["none_false_positive"] += 1
            elif c == 1:
                res["red_false_positive"] += 1
            elif c == 2:
                res["white_false_positive"] += 1
            elif c == 3:
                res["red_and_white_false_positive"] += 1
        
    # Check the missed detections
    for gt_i, (gt_pol, gt_class) in enumerate(zip(gt[0], gt[1])):
        # If found a match, skip as it was counted before
        if gt_i in gt_detected:
            continue
        # Extract ground truth values
        gt_class_id = GT_CLASSES[gt_class["gene expression"]]
        gt_mask = np.zeros(shape, dtype=np.uint8)
        rr, cc = skimage.draw.polygon(gt_pol["all_points_y"], gt_pol["all_points_x"])
        gt_mask[rr, cc] = 1
        gt_mask = gt_mask.astype(np.bool)
        gt_box = extract_bbox(gt_mask)

        # This was a missed detection, check the ground truth classification
        if gt_class_id == 0:
            res["none_false_negative"] += 1
        else:
            res["nuclei_false_negative"] += 1
            if gt_class_id == 1:
                res["red_false_negative"] += 1
            elif gt_class_id == 2:
                res["white_false_negative"] += 1
            elif gt_class_id == 3:
                res["red_and_white_false_negative"] += 1

    return res


def display_image(img, masks, boxes, classes, shape, ax):
    colours = []
    for c in classes:
        col = COLOUR_DICT[c]
        colours.append(col)

    viz.display_instances(img, boxes=boxes, masks=masks, class_ids=classes, class_names=["", "", "", "", ""],
                          colors=colours, figsize=(2048, 2048), show_bbox=False, ax=plt.gca())


############################################################
#  Test an architecture with a single model
############################################################
def test_single_architecture(model, ground_truth, img_dir, save_dir, do_display):
    total_results = {
        "avg_det_time": 0
    }

    # Test on each image
    for img_name, img_gt in ground_truth.items():
        # Read the image
        print(f"Running on {img_name}")
        img_path = os.path.join(img_dir, img_name)
        input_image = skimage.io.imread(img_path)

        # Run detection
        print("\tStarting detection")
        t = time.time()
        masks, boxes, classes = single_detect(model, input_image)
        t = time.time()-t
        print(f"\tDetection duration: {t}")
        total_results["avg_det_time"] += t/len(ground_truth.keys())
        # Evaluate the results
        if len(classes) == 0:
            masks = np.zeros((2048, 2048, 1), dtype=np.bool)
            boxes = np.zeros((4, 1), dtype=np.bool)
            classes = np.zeros(1, dtype=np.bool)
        results = evaluate_results(masks, boxes, classes, img_gt)

        # Print the results
        print(f"\tDetection results:")
        for stat, val in results.items():
            print(f"\t\tScore on {stat}: {val}")
            if type(val) == float:
                if stat not in total_results.keys():
                    total_results[stat] = 0
                total_results[stat] += val
            else:
                if stat not in total_results.keys():
                    total_results[stat] = []
                total_results[stat].appends(val)

        # Display the image
        fig = plt.figure(figsize=(22, 22))
        ax = fig.add_subplot(111)

        rgb_img_path = re.sub("/DatasetGeneChannels/", "/DatasetRGBChannels/", img_path)
        rgb_img = skimage.io.imread(rgb_img_path)
        display_image(rgb_img, masks, boxes, classes, input_image.shape, plt.gca())
        plt.tight_layout()

        if do_display and do_display is True:
            plt.show()

        if save_dir:
            save_path = os.path.join(save_dir, img_name)
            plt.savefig(save_path, dpi=96)
            pk.pickle()

    # Display total results
    print(f"\nFinal results:")
    for k, v in total_results.items():
        print(f"\tScore on {k}: {v}")


############################################################
#  Main
############################################################
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Test a given network against the ground truth.')
    parser.add_argument('--config', required=True,
                        metavar="<model_configuration>",
                        help='Name of the configuration of the architecture (standard, multi-channel, multi-model)')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights",
                        help="Path to weights .h5 file (for standard and multi-channel) or to "
                             "the folder containing the weights (for multi-model)")
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to test the model on')
    parser.add_argument("--img_dir", required=False,
                        metavar="/path/to/images/folder",
                        help="The path to the folder containing images to test the model on")
    parser.add_argument("--save_dir", required=False,
                        metavar="/path/to/save/folder",
                        help="The path to the save folder for the images produced")
    parser.add_argument("--do_display", required=False,
                        type=bool, default=False,
                        metavar="<display_result_img_flag>",
                        help="Flag to activate the display of the results")
    args = parser.parse_args()

    # Validate arguments
    assert args.image or args.img_dir, "Provide --image or --dir for the images ot analyse"
    assert args.config in ["standard", "multi-channel", "multi-model"], "Please use a supported architecture"

    # Load the ground truth
    if args.img_dir:
        dir_path = args.img_dir
    else:
        dir_path = os.path.split(args.image)[0]
    ground_truth = load_ground_truth(dir_path, args.image)

    # Test the model
    if args.config == "standard" or args.config == "multi-channel":
        print("Loading multi-channel architecture")
        config = StandardConfig()
        config.display()
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir="../logs")
        model.load_weights(args.weights, by_name=True)

        test_single_architecture(model, ground_truth, dir_path, args.save_dir, args.do_display)

    else:
        config = MultiNetworkNucleiConfig(), MultiNetworkGenesConfig()

    # Generate the model
