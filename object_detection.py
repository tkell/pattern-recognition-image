
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

import skimage.io as io
from skimage import util 

# Scikit Test data
from skimage import data
#image_data = data.coins()

# thor's test data
# this is coming in as floats, and
test_file_name = 'hexes_test.jpg'
image_data = io.imread(test_file_name, as_grey=True)
image_data = util.img_as_ubyte(image_data, force_copy=False)



# Do the math
elevation_map = sobel(image_data)
markers = np.zeros_like(image_data)
markers[image_data < 200] = 1
markers[image_data > 250] = 2
segmentation = morphology.watershed(elevation_map, markers)
segmentation = ndimage.binary_fill_holes(segmentation - 1)
labeled_image, _ = ndimage.label(segmentation)
image_label_overlay = label2rgb(labeled_image, image=image_data)


## SO!  Out of this, I need to find the centerpoints for each of those labels.
centroids_x = []
centroids_y = []
for region in regionprops(labeled_image):
    # These were reversed for some reason?
    centroids_x.append(int(region.centroid[1]))
    centroids_y.append(int(region.centroid[0]))

print len(centroids_y)

plt.imshow(image_data)
plt.scatter(centroids_x, centroids_y)
plt.show()
