import sys
import skimage.io as io
from skimage import util

import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from image_api import get_objects_as_boxes, create_button_data

if len(sys.argv) == 2:
    test_file_name = sys.argv[1]
    image_data = io.imread(test_file_name, as_grey=True)
    image_data = util.img_as_ubyte(image_data, force_copy=False)

    bounding_boxes = get_objects_as_boxes(image_data)
    button_data = create_button_data(bounding_boxes)

    plt.imshow(image_data)
    axis = plt.gca()

    for bounding_box in bounding_boxes:
        minr, minc, maxr, maxc = bounding_box
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='green', linewidth=1)
        axis.add_patch(rect)

    plt.show()  