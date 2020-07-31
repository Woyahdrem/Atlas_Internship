"""
Configuration for the cell Mask R-CNN

Written by Stefano Gatti

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 cell.py train --dataset=/path/to/cell/dataset --weights=coco

    # Train a new model starting from pre-trained COCO weights and using multiple masks per annotation
    python3 cell.py train --dataset=/path/to/cell/dataset --weights=coco --class_mode=multiple_masks

    # Train a new model starting from pre-trained COCO weights and using multiple masks per annotation
    python3 cell.py train --dataset=/path/to/cell/dataset --weights=coco --input_channels=genes
    
    # Resume training a model that you had trained earlier
    python3 cell.py train --dataset=/path/to/cell/dataset --weights=last
    
    # Train a new model starting from ImageNet weights
    python3 cell.py train --dataset=/path/to/cell/dataset --weights=imagenet
"""

import os
import sys
import json
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

# Import Mask RCNN CONFIG
from mrcnn.config import Config

NUMBER_OF_GENES = 2


############################################################
#  Configuration
############################################################

class CellConfigDefault(Config):
    """
    Configuration for training on the cell dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "cell"
    
    # The number of images per GPU depends on the size of the VRAM and the size of the images
    # Adjust as ceil( VRAM / max(img_size) )
    IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1500

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 15

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    # ROIs kept after tf.nn.top_k and before non-maximum suppression
    # May need adjusting due to high cell count
    PRE_NMS_LIMIT = 6000

    # ROIs kept after non-maximum suppression (training and inference)
    # May need adjusting due to high cell count
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # IMAGE_MIN_DIM is the size of the scaled shortest side
    # IMAGE_MAX_DIM is the maximum allowed size of the scaled longest side
    # May benefit from adjusting
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 500

    # Maximum number of ground truth instances  and final detections
    # These must definitely be higher
    MAX_GT_INSTANCES = 2000
    DETECTION_MAX_INSTANCES = 1000

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7
