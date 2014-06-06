import sys
import math

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

from skimage.feature import match_template

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
canny_segments = find_canny_edge_locations(image_data)
centroids_x, centroids_y, bounds = label_and_centerpoints(canny_segments)

size_threshold = 750
large_bounds = []
for bounding_box in bounds:
    minr, minc, maxr, maxc = bounding_box
    if (maxr - minr) * (maxc - minc) < size_threshold:
        continue
    else:
        large_bounds.append(bounding_box)

width_and_heights = []
final_boxes = []
for bounding_box in large_bounds:
    minr, minc, maxr, maxc = bounding_box
    width = maxc - minc
    height = maxr - minr

    # add the first one
    if not width_and_heights:
        width_and_heights.append((width, height))
        continue

    delta = 10
    for test_width, test_height in width_and_heights:
        if width > test_width - delta and width < test_width + delta:
            width_and_heights.append((width, height))
            final_boxes.append(bounding_box)
            break
print "We ended with  %d objects" % len(final_boxes)


plt.imshow(image_data)
axis = plt.gca()

plt.scatter(centroids_x, centroids_y)
for bounding_box in final_boxes:
    minr, minc, maxr, maxc = bounding_box
    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                              fill=False, edgecolor='green', linewidth=1)
    axis.add_patch(rect)

plt.show()  