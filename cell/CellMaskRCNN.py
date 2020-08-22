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

############################################################
#  Test
############################################################
def test(model, images_path, targets):
    t_r = 0
    t_w = 0
    t_rw = 0
    t_all = 0
    acc_r = 0
    acc_w = 0
    acc_rw = 0

    for img_i, (image_path, target) in enumerate(zip(images_path, targets)):

        # Run model detection
        print(f"Running on {image_path}")
        # Read image
        input_image = skimage.io.imread(image_path)
        image = skimage.io.imread(re.sub(r"/DatasetCellChannels/", "/DatasetRGBChannels/", image_path))
        # Detect cells
        r = model.detect([input_image], verbose=0)[0]
        """
        print(f"BBoxes: {r['rois']}\n\t{r['rois'].shape}\n\t{r['rois'].shape[0]}")
        print(f"Masks: {r['masks']}\n\t{r['masks'].shape}")
        print(f"Class Ids: {r['class_ids'].shape}\n\t{r['class_ids']}" +
              f"\n\tRed cells: {len([x for x in r['class_ids'] if x==1 or x==3])}" +
              f"\n\tWhite cells: {len([x for x in r['class_ids'] if x==2 or x==3])}")
        """
        print("----Counting classes----")
        all_red = len([x for x in r['class_ids'] if x == 1 or x == 3])
        all_white = len([x for x in r['class_ids'] if x == 2 or x == 3])
        red_and_white = len([x for x in r['class_ids'] if x == 3])
        cell_genes = len(r['class_ids'])
        t_r += all_red
        t_w += all_white
        t_rw += red_and_white
        t_all += cell_genes
        if target[0] == 0:
            if (all_red - red_and_white) == 0:
                acc_r += 1
            else:
                acc_r += 0
        else:
            acc_r += min((all_red - red_and_white) / target[0], 1)
        if target[1] == 0:
            if (all_white - red_and_white) == 0:
                acc_w += 1
            else:
                acc_w += 0
        else:
            acc_w += min((all_white - red_and_white) / target[1], 1)
        if target[2] == 0:
            if red_and_white == 0:
                acc_rw += 1
            else:
                acc_rw += 0
        else:
            acc_rw += min(red_and_white / target[2], 1)

        class_colors = {
            1: (.9, .0, .0),  # Red
            # 1: (.0, 1., 1.),  # Red but coloured as a negative

            2: (1., 1., 1.),  # White
            # 2: (.0, 1., 1.),  # White but coloured as a negative

            3: (1., .5, .0),  # Red and white
            # 3: (.9, .0, .0),  # Red and white but coloured red
            # 3: (1., 1., 1.),  # Red and white but coloured white

            4: (.0, 1., 1.)  # Negative
        }
        """
        colors = []
        for id in r["class_ids"]:
            colors.append(class_colors[id])
        viz.display_instances(image, boxes=r["rois"], masks=r["masks"], class_ids=r["class_ids"],
                              class_names=["background", "", "", ""  "", "negative"],
                              # class_names=["background", "red", "no red", "white", "no white"],
                              colors=colors,
                              figsize=(2048, 2048), show_bbox=False)
        """

    print(f"\n\n----Final report----")
    print(f"\tTotal red detections: {t_r} ({50 * (acc_r + acc_rw) / len(targets)}% accuracy)\n"
          f"\t\tExclusive red: {t_r - t_rw} ({100 * acc_r / len(targets)}% accuracy)")
    print(f"\tTotal white detections: {t_w} ({50 * (acc_w + acc_rw) / len(targets)}% accuracy)\n"
          f"\t\tExclusive white: {t_w - t_rw} ({100 * acc_w / len(targets)}% accuracy)")
    print(f"\tHybrid detections: {t_rw} ({100 * acc_rw / len(targets)}% accuracy)")
    print(
        f"\tTotal detections with genes: {t_all} ({100 * (acc_r + acc_w + acc_rw) / (3 * len(targets))}% accuracy)")


def detect(model, img):
    res = model.detect([img], verbose=0)[0]
    return res["masks"], res["rois"], res["class_ids"]


############################################################
#  Main
############################################################
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect gene expression in cells.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' (or 'detect' once implemented)")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/cell/dataset/",
                        help='Directory of the Cell dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image on which you detect cells')
    parser.add_argument("--dir", required=False,
                        metavar="/path/to/images/folder",
                        help="The path to the folder containing images to analyze")
    parser.add_argument('--input_channels', required=False,
                        default="rgb",
                        metavar="Channels of the input images",
                        help="Specifies the channels of the images: 'rgb' for standard channels and 'genes' for gene specific channels.")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "detect":
        assert args.image or args.dir, "Provide --image or --dir for the images to analyse"
    assert args.input_channels in ["rgb", "genes"], "--input_channels must be either 'rgb' or 'genes'"
    assert args.network in ["all", "RPN", "Heads"], "--network must be either 'all', 'RPN' or 'Heads'"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        if args.input_channels == "rgb":
            config = CellConfig()
        elif args.input_channels == "genes":
            class GeneConfig(CellConfig):
                IMAGE_CHANNEL_COUNT = 4
                MEAN_PIXEL = np.array([123.7, 116.8, 103.9, 100.0])


            config = GeneConfig()
    else:
        class InferenceConfig(CellConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Load images and targets
    if args.image:
        images = [args.image]
    else:
        images, targets = find_images(args.dir)

    # Train or evaluate
    if args.command == "train":
        train(model, args.input_channels)
    elif args.command == "detect":
        """
        if args.dir:
            count_instances(model, args.dir)
        else:"""
        detect(model=model, images_path=images, targets=targets)
    else:
        print(f"'{args.command}' is not recognized. \nUse 'train' (or 'detect' once implemented)".format(args.command))