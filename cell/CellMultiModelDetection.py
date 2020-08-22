"""
Mask R-CNN
Train on the Cell dataset

Written by Stefano Gatti

------------------------------------------------------------
merge command:
    --colour            - The colour of the genes of interest. Use once for each colour
    --image             - The path of the image to run detection on
    --model_dir         - The root directory containing the models, each in a folder named as its colour
    --cleanup_threshold - The IOU at which two detections from the same model are merged together
    --merge_threshold   - The IOU at which two detections from twho different models are merged together
    --pickle_file       - The name of the pickle file for the data of the detection
Example call:
python CellMultiModelDetection.py --colour=red --colour=white --image="..\\..\\DatasetGeneChannels\\train\\2019-04-03_RNAScope PAF OCT D380.lif [PAF OCT D380 FOXJ1 CFTR 1 1] Z4.jpg" --model_dir="..\\..\\..\\models\\cellSegmentation_Multi_Model"
python CellMultiModelDetection.py --colour=red --colour=white --dir="..\\..\\DatasetGeneChannels\\train" --model_dir="..\\..\\..\\models\\cellSegmentation_Multi_Model"
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
from datetime import datetime

sys.path.append("../")
from mrcnn import model as modellib, visualize as viz
from mrcnn import utils
from cell.CellGeneDetection import CellConfigGenes
from cell.CellNucleusDetection import CellConfigNuclei
from cell.StableMatching import stable_matching

DEFAULT_LOGS_DIR = os.path.join("../", "logs")
IOU_LOWER_BOUND = 0.9
CLASS_DICT = {
    "red": (1., .0, .0),
    "white": (1., 1., 1.)
}
RED_COLOURS = {
    1: (1., .0, .0),
    2: (0., 1., 1.)
}
WHITE_COLOURS = {
    1: (1., 1., 1.),
    2: (0., 1., 1.)
}
RED_AND_WHITE_COLURS = {
    (1, 1): (1., 0.5, 0.),
    (1, 2): (1., 0., 0.),
    (2, 1): (1., 1., 1.),
    (2, 2): (0., .5, 1.)
}
BB_YMIN = 0
BB_XMIN = 1
BB_YMAX = 2
BB_XMAX = 3


###############################################################################
#  Auxiliary functions
###############################################################################
def bbox(mask):
    """
    Computes the bounding box of a given mask
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, cmin, rmax, cmax


def compute_iou(mask1, bbox1, mask2, bbox2):
    """
    Computes the IoU between two masks
    Parameters:
        mask1   ->  Mask of the first object.
        bbox1   ->  Bounding box of the first object.
        mask2   ->  Mask of the second object.
        bbox2   ->  Bounding box of the second object.
    Return:
        IoU     ->  IoU value between the two masks
    """
    assert mask1.shape == mask2.shape, \
        f"\t\tThe two masks must have the same shape, but shapes {mask1.shape} and {mask2.shape} were found"

    # For faster computing, skip if there is no intersection between bounding boxes
    if not (bbox1[BB_XMIN] < bbox2[BB_XMAX] and bbox2[BB_XMIN] < bbox1[BB_XMAX]
            and bbox1[BB_YMIN] < bbox2[BB_YMAX] and bbox2[BB_YMIN] < bbox1[BB_YMAX]):
        return 0.

    intersection = np.sum(mask1 * mask2)

    if intersection == 0:
        return intersection

    union = np.sum(mask1 + mask2)
    res = intersection / union

    return res


###############################################################################
#  Merge collisions with IOU above a threshold in a single detection
###############################################################################
def get_detection_iou(masks, boxes):
    iou_list = {}

    for i in range(masks.shape[-1]):
        for j in range(i + 1, masks.shape[-1]):
            m1 = masks[:, :, i]
            m2 = masks[:, :, j]
            b1 = boxes[i]
            b2 = boxes[j]
            iou = compute_iou(m1, b1, m2, b2)
            if iou == 0:
                continue
            iou_list[(i, j)] = iou
        print(
            f"\t\tCompleted cycle {i + 1}/{masks.shape[-1]}\tCollisions detected: {len(iou_list.values())}", end="\r")

    return iou_list


def cleanup_detection(iou_threshold, masks, boxes, classes):
    """
    Parameters:
        iou_threshold:  ->  Lowest IoU for two masks to be considered of the same nucleus.
        masks:          ->  Numpy array [H,W,C] containing the masks of the detection.
        boxes:          ->  Numpy array [4,C] containing the bounding boxes of the detection.
        classes:        ->  Numpy array [C,] containing the classes of the detection.
    Returns:
        ->  Numpy array [H,W,C*] containing the masks of the merged detection.
        ->  Numpy array [4,C*] containing the bounding boxes of the merged detection.
        ->  Numpy array [C*,] containing the classes of the merged detection.
        ->  Float containing the max value of the IoUs between all the masks.
    """
    # Compute the IoU list
    iou = get_detection_iou(masks, boxes)
    if not iou:
        print(f"\n\tNo collisions here!")
        return masks, boxes, classes, 0.
    if max(iou.values()) < iou_threshold:
        print(f"\n\tNothing to merge, returning data as-is")
        return masks, boxes, classes, max(iou.values())

    # Prepare the index lists
    merges = []
    non_merges = list(range(masks.shape[-1]))
    max_iou = copy.deepcopy(iou)
    # Find the collisions
    for (i, j), val in iou.items():
        if val < iou_threshold:
            continue
        print(f"\tMask {i} and {j} are in collision (IOU = {val})")
        # Find if i or j have already appeared in the merges
        found = False
        idx = -1
        for list_idx, list_merges in enumerate(merges):
            if i in list_merges or j in list_merges:
                found = True
                triple = True
                idx = list_idx
                break
        # Add i and j to the merges
        if found:
            merges[idx].extend([i, j])
        else:
            merges.extend([[i, j]])
        # Remove i and j from the non merges list
        if i in non_merges:
            non_merges.remove(i)
        if j in non_merges:
            non_merges.remove(j)
        max_iou.pop((i, j), None)
    if len(merges) == 0:
        print(f"\tNothing to merge, returning data as-is")
        return masks, boxes, classes
    print(f"\tMasks to merge:\n\t{merges}")
    # Create the new data structures
    merged_masks = masks[:, :, non_merges]
    merged_boxes = boxes[non_merges]
    merged_classes = classes[non_merges]
    max_iou = max(max_iou.values())
    # Append the merged data
    for idx, merges_list in enumerate(merges):
        print(f"\tMerging {len(merges_list)} elements: {merges_list}")
        # Base case
        i = merges_list[0]
        m = masks[:, :, i]
        b = boxes[i]
        c = classes[i]
        # Remove duplicates and the first element (used for the base case)
        idx_list = list(set(merges_list[1:]))
        for new_idx in idx_list:
            # Gather the data to merge
            new_m = masks[:, :, new_idx]
            new_c = classes[new_idx]
            # Merge the data
            m = m + new_m
            b = bbox(m)
            if c == 1 or new_c == 1:
                c = 1
            else:
                c = 2
        merged_masks = np.dstack((merged_masks, m))
        merged_boxes = np.vstack((merged_boxes, b))
        merged_classes = np.hstack((merged_classes, c))
    # Return the results
    return merged_masks, merged_boxes, merged_classes, max_iou


###############################################################################
#  Merge two different detections in a single one
###############################################################################
def compute_overlap(masks_1, boxes_1, masks_2, boxes_2):
    """
    Function to compute the IoU between the two sets of masks
    Parameters:
        masks_1:    ->  Numpy array [W,H,C1] containing the masks of the first detection.
        boxes_1:    ->  Numpy array [4,C1] containing the bounding boxes of the first detection.
        masks_2:    ->  Numpy array [W,H,C2] containing the masks of the second detection.
        boxes_2:    ->  Numpy array [4,C2] containing the bounding boxes of the second detection.
    Return:
        res:        ->  Dictionary containing the IoU between all pairs of masks
    """
    res = {}

    # Examine all masks from the base masks
    for i in range(masks_1.shape[-1]):
        m1 = masks_1[:, :, i]
        b1 = boxes_1[i]
        # Examine the base mask against all other ones
        for j in range(masks_2.shape[-1]):
            m2 = masks_2[:, :, j]
            b2 = boxes_2[j]

            # For faster computing, skip if there is no intersection between bounding boxes
            if not (b1[BB_XMIN] < b2[BB_XMAX] and b2[BB_XMIN] < b1[BB_XMAX]
                    and b1[BB_YMIN] < b2[BB_YMAX] and b2[BB_YMIN] < b1[BB_YMAX]):
                continue

            inters = np.sum(m1 * m2)
            if inters == 0:
                # If the intersection is 0, there is no overlap
                continue

            union = np.sum(m1 + m2)
            iou = inters / union

            print(f"\tFound a collision between mask {i} and mask {j} with IOU equal to {iou}", end="\r")
            res[(i, j)] = iou

    # Return the dictionary
    return res


def find_matching(masks_1, masks_2, iou, threshold):
    # Find the matching using a stable marriage
    collisions_1, collisions_2 = stable_matching(iou, threshold)

    # Find the elements not in collision
    no_collisions_1 = [i for i in list(range(masks_1.shape[-1])) if i not in collisions_1]
    no_collisions_2 = [i for i in list(range(masks_2.shape[-1])) if i not in collisions_2]

    # Return the output lists
    return collisions_1, no_collisions_1, collisions_2, no_collisions_2


def merge_detections(masks_1, boxes_1, classes_1, masks_2, boxes_2, classes_2, threshold=0.6):
    """
    Function to merge two detections together.
    Parameters:
        masks_1:    ->  Numpy array [W,H,C1] containing the masks of the first detection.
        boxes_1:    ->  Numpy array [4,C1] containing the bounding boxes of the first detection.
        classes_1:  ->  Numpy array [C1,X] containing the classification of the first detection.
        masks_2:    ->  Numpy array [W,H,C2] containing the masks of the second detection.
        boxes_2:    ->  Numpy array [4,C2] containing the bounding boxes of the second detection.
        classes_2:  ->  Numpy array [C2,] containing the classification of the second detection.
        threshold:  ->  Lowest IoU for two masks to be considered of the same nucleus.
    Return:
        masks:      ->  Numpy array [W,H,C3] containing the masks of the merged detection.
        boxes:      ->  Numpy array [4,C3] containing the bounding boxes of the merged detection.
        classes:    ->  Numpy array [C3,X+1] containing the classification of the merged detection.
    """

    # If the first detection is None, return the second one without any further operation
    if masks_1 is None and boxes_1 is None and classes_1 is None:
        print(f"\tFirst detection, noting to merge")
        return masks_2, boxes_2, classes_2

    # We must check that the shapes of the masks is the same for merging
    assert masks_1.shape[:-1] == masks_2.shape[:-1], \
        f"\tMask mismatch: cannot merge masks of shape {masks_1.shape[:-1]} with masks of shape {masks_2.shape[:-1]}"

    # The classes need to be 2D arrays
    if len(classes_1.shape) == 1:
        print(f"\tAdding a dimension to classes_1")
        classes_1 = np.expand_dims(classes_1, axis=1)
    if len(classes_2.shape) == 1:
        print(f"\tAdding a dimension to classes_2")
        classes_2 = np.expand_dims(classes_2, axis=1)

    # Find elements in masks_2 in collision with elements in masks_1
    iou_dict = compute_overlap(masks_1, boxes_1, masks_2, boxes_2)

    # Find the collisions
    coll_1, no_coll_1, coll_2, no_coll_2 = find_matching(masks_1, masks_2, iou_dict, threshold)
    print(f"\tCollisions detected: {len(coll_1)}")

    # The classes need to be 2-dimensional for stacking
    if len(classes_1.shape) == 1:
        classes_1 = np.expand_dims(classes_1, axis=1)
    if len(classes_2.shape) == 1:
        classes_1 = np.expand_dims(classes_2, axis=1)

    # Merge the elements with collisions
    merged_masks = masks_1[:, :, coll_1] + masks_2[:, :, coll_2]
    merged_classes = np.hstack((classes_1[coll_1], classes_2[coll_2]))

    assert len(coll_1) == len(coll_2) == merged_masks.shape[-1], f"{coll_1}  {coll_2} {merged_masks.shape[-1]}"

    # The boxes are more difficult, as we need to apply a function to each layer
    merged_boxes = np.zeros((merged_masks.shape[-1], 4))
    for i in range(len(coll_1)):
        merged_boxes[i, :] = bbox(merged_masks[:, :, i])

    # Merge the elements without detections
    if len(no_coll_1) > 0:
        # Create the additions
        added_masks = masks_1[:, :, no_coll_1]
        added_boxes = boxes_1[no_coll_1]
        added_classes = np.hstack((classes_1[no_coll_1],
                                   np.ones((len(no_coll_1), 1), dtype=classes_1.dtype) * 2))
        # Stack the additions
        merged_masks = np.dstack((merged_masks, added_masks))
        merged_boxes = np.vstack((merged_boxes, added_boxes))
        merged_classes = np.vstack((merged_classes, added_classes))
    if len(no_coll_2) > 0:
        # Create the additions
        added_masks = masks_2[:, :, no_coll_2]
        added_boxes = boxes_2[no_coll_2]
        added_classes = np.hstack((np.ones((len(no_coll_2), classes_1.shape[1]), dtype=classes_1.dtype) * 2,
                                   classes_2[no_coll_2]))
        # Stack the additions
        merged_masks = np.dstack((merged_masks, added_masks))
        merged_boxes = np.vstack((merged_boxes, added_boxes))
        merged_classes = np.vstack((merged_classes, added_classes))

    return merged_masks, merged_boxes, merged_classes


###############################################################################
#  Execute multiple detections
###############################################################################
def detect(model, weights_dir, colours, img, clean_threshold=0.7, merge_threshold=0.7, weight_name="mask_rcnn_cell.h5"):
    """
    Execute detection on an image with all networks and merge the results.
    Params:
        model           ->  Shared architecture of the models.
        weights_dir     ->  Path of the folder containing the weights of the specialized networks.
        colours         ->  List of subfolders of weights_dir and staining colours to run detection on.
        img             ->  Image to run detection on.
        clean_threshold ->  Float between 0 and 1 above which two masks are considered of the same nuclei in the same detection
        merge_threshold ->  Float between 0 and 1 above which two masks are considered of the same nuclei in different detections
        weight_name     ->  Name of the .h5 file containing the weights of the specialized network.
    :return:
        total_masks     ->  Numpy array [W,H,C] containing the final detection masks.
        total_boxes     ->  Numpy array [4,C] containing the final detection bounding boxes.
        total_classes   ->  Numpy array [C,] containing the final detection classes.
    """
    # Prepare the data structure
    total_masks = None
    total_boxes = None
    total_classes = None

    # Cycle on each colour/model
    for colour in colours:

        # Load the specialized model
        weights_path = os.path.join(weights_dir, colour, weight_name)
        model.load_weights(weights_path, by_name=True)
        print(f"\n----Color {colour}----\nLoading weights from {weights_path}")

        # Activate the network
        print(f"Activating neural network...")
        r = model.detect([img], verbose=0)[0]
        new_masks = r["masks"]
        new_boxes = r["rois"]
        nwe_classes = r["class_ids"]
        print(f"Done")

        # Merge the multiple detection of nuclei
        print(f"Cleaning up the detection (Merge threshold: {clean_threshold})")
        masks, boxes, classes, _ = cleanup_detection(clean_threshold, masks, boxes, classes)
        print(f"Done")

        # Merge the new detection with the total detection
        print(f"Merging the current detection with the others (Merge threshold: {merge_threshold})")
        total_masks, total_boxes, total_classes = merge_detections(total_masks, total_boxes, total_classes,
                                                                   masks, boxes, classes,
                                                                   threshold=merge_threshold)
        print(f"Done")

    # Finally, return the final detection
    return total_masks, total_boxes, total_classes

