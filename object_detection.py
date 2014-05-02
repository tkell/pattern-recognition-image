
# From here: http://scikit-image.org/docs/dev/auto_examples/applications/plot_coins_segmentation.html
# the goal is to get the center points of the segmentation back

import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

import numpy as np
from scipy import ndimage
from skimage.filter import sobel
from skimage import morphology
from skimage.color import label2rgb

from skimage.measure import regionprops


# Test data
from skimage import data
image_data = data.coins()

# Do the math
elevation_map = sobel(image_data)
markers = np.zeros_like(image_data)
markers[image_data < 30] = 1
markers[image_data > 150] = 2
segmentation = morphology.watershed(elevation_map, markers)
segmentation = ndimage.binary_fill_holes(segmentation - 1)
labeled_image, _ = ndimage.label(segmentation)
image_label_overlay = label2rgb(labeled_image, image=image_data)


## SO!  Out of this, I need to find the centerpoints for each of those labels.
# Oh look.
centroids_x = []
centroids_y = []
for region in regionprops(labeled_image):
    # These were reversed for some reason?
    centroids_x.append(int(region.centroid[1]))
    centroids_y.append(int(region.centroid[0]))

plt.imshow(image_label_overlay)
plt.scatter(centroids_x, centroids_y)
plt.show()
