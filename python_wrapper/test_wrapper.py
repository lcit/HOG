import sys
import os
sys.path.append( os.getcwd() )
import HOG_module
import numpy as np
from PIL import Image

image = np.array(Image.open("../img/person.JPG").convert(mode='L'))
print image

class GradientType:
    SIGNED=0
    UNSIGNED=1
class NormalizationType:
    none=0
    L1norm=1
    L1sqrt=2
    L2norm=3
    L2hys=4

blocksize = 64
cellsize = 32
stride = 64
binning = 9
grad_type = GradientType.SIGNED
norm_type = NormalizationType.none   
    
hog_hist = HOG_module.HOG_func(blocksize, cellsize, stride, binning, grad_type, norm_type, image)
print hog_hist
