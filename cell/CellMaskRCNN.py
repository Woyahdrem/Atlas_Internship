"""
Mask R-CNN
Train on the Cell dataset

Written by Stefano Gatti

------------------------------------------------------------

"""

import os
import re
import sys
import json
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
assert len(ROOT_DIR) != 0, "Please specify the root directory (variable ROOT_DIR)"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find the local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils, visualize as viz

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################
class CellConfig(Config):
    """Configuration for training on the cell dataset.
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

    # Number of classes (including background)
    # Find out how to use multiple labels
    NUM_CLASSES = 1 + 3  # Background + ( red + white + red and white )

    # Length of square anchor side in pixels
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 512

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

    # Number of color channels per image
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    # Right now we are using the usual RGB channels
    IMAGE_CHANNEL_COUNT = 3

    # Image mean (RGB)
    # Must have length equal to IMAGE_CHANNEL_COUNT
    # Values could depend on brightness of layer
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 500

    # Maximum number of ground truth instances  and final detections
    # These must definitely be higher
    MAX_GT_INSTANCES = 1000
    DETECTION_MAX_INSTANCES = 500

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7


############################################################
#  Dataset
############################################################
class CellDatasetDefault(utils.Dataset):

    def load_cell(self, dataset_dir, subset):
        """Load a subset of the Cell dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have multiple classes, one for each gene
        # What is the difference between source and class_name?
        self.add_class("cell", 1, "red")
        self.add_class("cell", 2, "white")
        self.add_class("cell", 3, "red and white")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "via_mask_annotations_json.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # Skip unannotated images
        annotations = [a for a in annotations if a['regions']]

        # Add the images
        for a in annotations:
            # We are using Bounding Boxes instead of Masks
            # To use the masks we will convert the boxes in polygons and use them as masks
            polygons = [r['shape_attributes'] for r in a['regions']]

            # Since we are using multiple classes, we also need to know what class each annotation belongs to
            classes = [r['region_attributes'] for r in a['regions']]

            # Next, we need to load the image path and the image size
            # if the dataset becomes too big, having the values directly in the json becomes necessary
            image_path = os.path.join(dataset_dir, a['filename'])
            img = skimage.io.imread(image_path)
            height, width = img.shape[:2]

            self.add_image(
                "cell",
                image_id=a["filename"],
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                classes=classes)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a cell dataset image, delegate to parent class
        info = self.image_info[image_id]
        if info["source"] != "cell":
            return super(self.__class__, self).load_mask(image_id)

        # Compute the number of non-negative cells
        num_cells = len(info["classes"]) - len([c for c in info["classes"] if c["gene expression"] == "negative"])

        # Convert the BBox to a bitmap mask of shape
        # [height, width, instance_count]
        # Right now we are converting the Bounding Box into a rectangular mask
        mask = np.zeros([info["height"], info["width"], num_cells], dtype=np.uint8)
        class_ids = np.zeros(num_cells)

        # Set the masks for each instance
        # At the same time, set class_ids to the corresponding class
        # If the current annotation corresponds to a negative, skip it
        i = 0
        for p, c in zip(info["polygons"], info["classes"]):
            if c["gene expression"] == "negative":
                continue
            rr, cc = skimage.draw.polygon(p["all_points_y"], p["all_points_x"])
            mask[rr, cc, i] = 1
            if c["gene expression"] == "red":
                class_ids[i] = 1
            elif c["gene expression"] == "white":
                class_ids[i] = 2
            elif c["gene expression"] == "red and white":
                class_ids[i] = 3
            else:
                print(f"Class not recognized: {c['gene expression']} in image {image_id}")
            i += 1

        # Return mask and class ID array
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cell":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################
def train(model, dataset, config):
    """Train the model."""

    # Training dataset
    dataset_train = CellDatasetDefault()
    dataset_train.load_cell(dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CellDatasetDefault()
    dataset_val.load_cell(dataset, "val")
    dataset_val.prepare()

    # Select the layers to train
    layers = "heads"

    # Define the augmentation of for the dataset
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Rotate((-45, 45)),
        iaa.ScaleX((0.8, 1.2)),
        iaa.ScaleY((0.8, 1.2))
    ])

    # Finally, train the model
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=100,
                layers=layers,
                augmentation=augmentation)
