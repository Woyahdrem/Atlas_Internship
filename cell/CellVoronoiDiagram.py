"""
Use output of detection to divide an image into cells using a voronoi diagram

Written by Stefano Gatti
"""

# General imports
import os
import sys
import itertools
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.spatial import Voronoi, voronoi_plot_2d

# Local imports
sys.path.append("../")
from mrcnn import visualize as viz


############################################################
#  Colours
############################################################
RED_WHITE_DICT = {
    (1, 1): (1., .5, .0),
    (1, 2): (1., .0, .0),
    (2, 1): (1., 1., 1.),
    (2, 2): (0., .5, .1),
}

def get_colours(classes, col_dict):
    colours = []
    for c in classes:
        comb = tuple(c)
        col = col_dict[comb]
        colours.append(col)
    return colours


############################################################
#  Loading
############################################################
def load_data(filepath):
    masks, boxes, classes, image = pk.load(open(filepath, "rb"))
    return masks, boxes, classes, image


############################################################
#  Centroids
############################################################
def compute_centroids(masks):
    # Compute the centroid for each mask
    centroids = []
    for i in range(masks.shape[-1]):
        mask = masks[:, :, i]
        c = ndimage.measurements.center_of_mass(mask)
        if np.isnan(np.sum(c)):
            print(np.any(mask))
            print(mask)
            print(c)
        centroids.append(c)
    centroids = np.array(centroids)
    return centroids


############################################################
#  Voronoi Diagram
############################################################
def generate_vor_diagram(centroids):
    # Generate the voronoi diagram given the centroids of the nuclei

    vor = Voronoi(np.flip(centroids))

    return vor


############################################################
#  Main
############################################################
dir = "D:/Documents/Università/Atlas/Cell-Mask-RCNN/Mask_RCNN-master/logs"

for file in os.listdir(dir):
    if file.endswith(".pk"):
        print(f"\n----{file}----")
        masks, boxes, classes, img = pk.load(open(os.path.join(dir, file), "rb"))

        idx = list(range(masks.shape[-1]))
        rem = []
        for i in range(masks.shape[-1]):
            if not np.any(masks[:, :, i]):
                idx.remove(i)
                rem.append(i)
        if len(rem) > 0:
            print(f"Detections {rem} have no mask, removing")
            masks = masks[:, :, idx]
            boxes = boxes[idx]
            classes = classes[idx]

        colours = get_colours(classes, RED_WHITE_DICT)

        centr = compute_centroids(masks)
        vor = generate_vor_diagram(centr)

        fig = plt.figure(figsize=(2048, 2048))
        ax = fig.add_subplot(111)

        voronoi_plot_2d(vor, point_size=10, ax=ax, line_colors=(.0, .5, 1.), show_vertices=False, show_points=False, line_width=1)

        for x, y, c in zip(centr[:, 1], centr[:, 0], colours):
            plt.scatter(x, y, 10, np.expand_dims(c, axis=0))

        viz.display_instances(img, boxes=boxes, masks=masks, class_ids=classes[:,0], class_names=["", "", "", "", ""],
                              colors=colours, figsize=(2048, 2048), show_bbox=False, ax=plt.gca())

        plt.show()




