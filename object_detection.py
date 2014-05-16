
# From here: http://scikit-image.org/docs/dev/auto_examples/applications/plot_coins_segmentation.html
# the goal is to get the center points of the segmentation back
import sys

import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import numpy as np
from skimage.filter import canny, denoise_bilateral
from scipy import ndimage

from skimage.filter import sobel
from skimage import morphology
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.filter import rank

import skimage.io as io
from skimage import util

def label_and_centerpoints(cleaned_data):
    labeled_image, _ = ndimage.label(cleaned_data)
    centroids_x = []
    centroids_y = []
    bounds = []
    for region in regionprops(labeled_image):
        # These were reversed for some reason?
        centroids_x.append(int(region.centroid[1]))
        centroids_y.append(int(region.centroid[0]))
        bounds.append(region.bbox)
    print "We found %d objects" % len(centroids_y)
    return centroids_x, centroids_y, bounds

def find_watershed_locations(image_data):
    elevation_map = sobel(image_data)
    markers = np.zeros_like(image_data)
    markers[image_data < 30] = 1
    markers[image_data > 150] = 2
    segmentation = morphology.watershed(elevation_map, markers)
    cleaned_data = ndimage.binary_fill_holes(segmentation - 1)
    return cleaned_data

def find_canny_edge_locations(image_data):
    edges = canny(image_data / 255.0, low_threshold=0.0, high_threshold=0.0)
    fill_data = ndimage.binary_fill_holes(edges)
    cleaned_data = morphology.remove_small_objects(fill_data, 150)
    return cleaned_data

if len(sys.argv) == 2:
    test_file_name = sys.argv[1]
    image_data = io.imread(test_file_name, as_grey=True)
    image_data = util.img_as_ubyte(image_data, force_copy=False)
else:
    from skimage import data
    image_data = data.coins()



# So far, it looks like canny edge detection is the best
# I feel like I want to turn up the edgyness of the edge detection?
# I think I need to talk to some people about this.  Next friday!

# If I take my user input and go looking for button size, can I do that?
# I think so:  move out from the button point, find things of a certain similarity, 
# then fuzz them out a bit

# or can I tile these things / look for negative space?
# 

canny_segments = find_canny_edge_locations(image_data)
centroids_x, centroids_y, bounds = label_and_centerpoints(canny_segments)
plt.imshow(image_data)
plt.scatter(centroids_x, centroids_y)
axis = plt.gca()

size_threshold = 750
for bounding_box in bounds:
    minr, minc, maxr, maxc = bounding_box
    if (maxr - minr) * (maxc - minc) < size_threshold:
        continue
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='green', linewidth=1)
    axis.add_patch(rect)

plt.show()