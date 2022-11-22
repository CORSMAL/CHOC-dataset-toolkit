"""
The background pixels in the NOCS images are not [0,0,0]. This is code to fix that.

It seems that the background pixels are always [13,13,13] or [14,14,14], but we haven't verified that.
Therefore we choose to segment the foreground from background using Otsu's algorithm:
We use https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html#:~:text=Otsu's%20Binarization,determines%20it%20automatically.

Otsu works well, since there is a clear separation between foreground/background, but we found it is not perfect. I.e. sometimes, 
it will remove some NOCS pixels.
"""
import cv2
import argparse
import numpy as np
import utils
import os

# Parsing arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument('--choc_dir', type=str, help="Path to the CHOC dataset.")
parser.add_argument('--image_index', type=str, help="Image index in 6-digit string format.", default="000001")
args = parser.parse_args()

# Get a NOCS image
nocs_image_path = os.path.join(args.choc_dir, "mixed-reality", "nocs", utils.image_index_to_batch_folder(args.image_index), "{}.png".format(args.image_index))
nocs_image = cv2.imread(nocs_image_path)[:,:,::-1] # load as RGB

# Visualise the problem [zoom in on background to see the pixel values]
cv2.imshow("problematic nocs background", nocs_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Fix the problem
print("unique values in nocs:", np.unique(nocs_image))
nocs_image_fixed = utils.fix_background_nocs(nocs_image)
print("unique values in nocs fixed:", np.unique(nocs_image_fixed))

# Visualise the fixed nocs [zoom in on background to see the pixel values]
cv2.imshow("fixed nocs background", nocs_image_fixed)
cv2.waitKey(0)
cv2.destroyAllWindows()



