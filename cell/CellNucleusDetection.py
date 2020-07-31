"""
Mask R-CNN
Train and detect on the Cell dataset for nucleus detection

Written by Stefano Gatti

------------------------------------------------------------

Usage: run from command line
Parameters:
        # General
    command             ->  Either 'train' for the training, or 'test' for testing the network.
        # Training
    --dataset           ->  The folder of the dataset to use for training. Must be divided in 'train' and 'val' subfolders.
    --weights           ->  The location of the .h5 file to load a previous model or 'coco' to use the COCO model.
    --epochs            ->  The number of epochs to run training.
    --logs              ->  The folder where logs will be saved. Defaults to ./logs.
        # Testing
    --image             ->  The image to run detection on during testing. Use this or '--dir'.
    --dir               ->  The folder containing the images to run detection on during testing. Use this or '--image'.
    --channel_number    ->  The number of channels of the input images. Defaults to 3.
    --display_image     ->  Whether to display the result of testing ot not.
Examples:
        # Training
    python CellNucleusDetection.py train --dataset=../datasets/DatasetGeneChannels --weights=../models/cellSegmentation_RPN_only/mask_rcnn_cell_0010.h5 --epochs=20
        # Testing
    python CellNucleusDetection.py test --image="D:/Documents/Università/Atlas/Cell-Mask-RCNN/Mask_RCNN-master/DatasetRGBChannels/train/2019-04-03_RNAScope PAF OCT D380.lif [PAF OCT D380 FOXJ1 CFTR 1 1] Z6.jpg" --weights="../../../models/cellSegmentation_RPN_only/mask_rcnn_cell_0010.h5"
"""

# Generic imports
import os
import re
import sys
import json
import time
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa

# Define the root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)

# Import local files
from cell.CellConfiguration import CellConfigDefault
from mrcnn import model as modellib, utils, visualize as viz

############################################################
#  Global variables
############################################################
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configuration
############################################################
class CellConfigNuclei(CellConfigDefault):
    """
    Specialization of the config file for the network that needs to detect the nuclei.
    In the future, this will include a constructor method to define dynamically the channel count.
    """
    # Number of classes (including background)
    # We only need to detect the nuclei, so only 1 additional class
    NUM_CLASSES = 1 + 1

    # Number of color channels per image
    # Changing this requires other changes in the code. See the WIKI for more
    # details: https://github.com/matterport/Mask_RCNN/wiki
    # Right now we are using images with 2 genes
    IMAGE_CHANNEL_COUNT = 1 + 2

    # Image mean (RGB)
    # Must have length equal to IMAGE_CHANNEL_COUNT
    # Values could depend on brightness of layer
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])


############################################################
#  Dataset
############################################################
class CellDatasetNuclei(utils.Dataset):

    def load_cell(self, dataset_dir, subset):
        """Load a subset of the Cell dataset.
         dataset_dir: Root directory of the dataset.
         subset: Subset to load: train or val
         """
        # Add classes.
        # We have only one class: the nucleus of a cell.
        self.add_class("cell", 1, "nucleus")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load the annotations form the json file
        annotations = json.load(open(os.path.join(dataset_dir, "via_mask_annotations_json.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # Skip unannotated images
        annotations = [a for a in annotations if a['regions']]

        # Analyze the annotations to add images and ground truths.
        for a in annotations:
            # Load the polygons in [[x1, y1], ...] format
            polygons = [r['shape_attributes'] for r in a['regions']]

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
                polygons=polygons)

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

        # Compute the number of nuclei
        num_nuclei = len(info["polygons"])

        # Define the class ids
        class_ids = np.ones(num_nuclei, dtype=np.uint8)

        # Set the masks for each instance
        mask = np.zeros([info["height"], info["width"], num_nuclei], dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            rr, cc = skimage.draw.polygon(p["all_points_y"], p["all_points_x"])
            mask[rr, cc, i] = 1

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
def train(model, dataset, config, epochs):
    """Train the model."""

    # Training dataset
    dataset_train = CellDatasetNuclei()
    dataset_train.load_cell(dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CellDatasetNuclei()
    dataset_val.load_cell(dataset, "val")
    dataset_val.prepare()

    # Select the layers to train
    layers = r"(conv1)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)"

    # Define the augmentation of for the dataset
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Rotate((-45, 45)),
        iaa.ScaleX((0.7, 1.3)),
        iaa.ScaleY((0.7, 1.3))
    ])

    # Finally, train the model
    print(f"\n----Beginning training----\n")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,
                layers=layers,
                augmentation=augmentation)


############################################################
#  Testing
############################################################
def test(model, images_path, targets, do_display):
    tot_n = 0
    acc_n = 0
    time_n = 0

    for img_i, (image_path, target) in enumerate(zip(images_path, targets)):
        # Run model detection
        print(f"Running on {image_path}")
        img_start = time.time()

        # Read image
        input_image = skimage.io.imread(image_path)

        # Detect cells
        r = model.detect([input_image], verbose=0)[0]

        # Print results of current image
        print("\tTest results")
        nuclei_num = len(r['class_ids'])
        tot_n += nuclei_num
        if target == 0:
            if nuclei_num == 1:
                nuclei_acc = 1
            else:
                nuclei_acc = 0
        else:
            nuclei_acc = nuclei_num / target
        acc_n += nuclei_acc
        print(f"\t\tNuclei found: {nuclei_num}\n\t\tAccuracy: {100 * nuclei_acc}%")

        # Detection time
        runtime = time.time() - img_start
        time_n += runtime
        print(f"\tDetection time: {runtime}")

        # Display the results
        if do_display:
            image = skimage.io.imread(re.sub(r"/DatasetCellChannels/", "/DatasetRGBChannels/", image_path))
            colours = []
            for _ in r['class_ids']:
                colours.append((0., .5, 1.))
            viz.display_instances(image, boxes=r["rois"], masks=r["masks"], class_ids=r["class_ids"],
                                  class_names=["background", ""], colors=colours, figsize=(2048, 2048), show_bbox=False)

    print(f"\n\n----Final report----")
    print(f"\tTotal nuclei detected: {tot_n}")
    print(f"\tAverage accuracy: {acc_n / len(targets)}")
    print(f"\tAverage detection time: {time_n / len(targets)}")


############################################################
#  Main
############################################################
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect cell nucleus in an image.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'test'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/cell/dataset/",
                        help='Directory of the Cell dataset for training.')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'.")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/).')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image on which you detect nuclei.')
    parser.add_argument('--dir', required=False,
                        metavar="path or URL to image",
                        help='Directory containing images on which you detect nuclei.')
    parser.add_argument('--channel_number', required=False,
                        default=3, type=int,
                        metavar="Number of channels in the input images",
                        help="Specifies the number of channels in the image.")
    parser.add_argument('--display_image', required=False,
                        default=False, type=bool,
                        metavar="<displayTestingResults>",
                        help="Specifies whether the results of the testing are visualized with an image.")
    parser.add_argument('--epochs', required=False,
                        default=10, type=int,
                        metavar="<training_epochs>",
                        help="Number of epochs for training the model.")
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "test":
        assert args.image or args.dir, "Provide --image or --video to apply color splash"

    # Load the configuration
    if args.command == "train":
        config = CellConfigNuclei()
    else:
        class InferenceConfig(CellConfigNuclei):
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

    # Train or test
    if args.command == "train":
        train(model=model, dataset=args.dataset, config=config, epochs=args.epochs)
    elif args.command == "test":
        # Load the image and corresponding ground truths
        if args.image:
            # Load the annotation file
            dir = os.path.split(args.image)[0]
            print(f"\tLoading images from the folder {dir}")
            a = json.load(open(os.path.join(dir, "via_mask_annotations_json.json")))
            a = list(a.values())

            targets = []
            for file in a:
                if a["filename"] != os.path.split(args.image)[1]:
                    continue
                n = len(file["regions"])
                targets.append(n)
            images = [args.image]
        else:
            # Load the annotation file
            print(f"\tLoading images from the folder {dir}")
            a = json.load(open(os.path.join(dir, "via_mask_annotations_json.json")))
            a = list(a.values())

            # Load the image names
            images = [os.path.join(dir, x["filename"]) for x in a if x["regions"]]
            print(f"\tImages found: ")
            print("\n".join([os.path.split(x)[1] for x in images]))

            targets = []
            for file in a:
                n = len(file["regions"])
                targets.append(n)

        test(model=model, images_path=images, targets=targets, do_display=args.display_image)
