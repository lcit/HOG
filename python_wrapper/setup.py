from distutils.core import setup, Extension

# define the extension module
HOG_module = Extension('HOG_module', sources=['HOG_module.cpp', '../HOG.cpp'], extra_compile_args=['-std=c++14', '-O2'], extra_link_args=['-fopenmp'], include_dirs=['..','/usr/local/include/opencv','/usr/local/include'], library_dirs=['.'], libraries=['opencv_videostab','opencv_videoio','opencv_video','opencv_superres','opencv_stitching','opencv_shape','opencv_photo','opencv_objdetect','opencv_ml','opencv_imgproc','opencv_imgcodecs','opencv_highgui','opencv_flann','opencv_features2d','opencv_cudev','opencv_cudawarping','opencv_cudastereo','opencv_cudaoptflow','opencv_cudaobjdetect','opencv_cudalegacy','opencv_cudaimgproc','opencv_cudafilters','opencv_cudafeatures2d','opencv_cudacodec','opencv_cudabgsegm','opencv_cudaarithm','opencv_core','opencv_calib3d'])

# run the setup
setup(ext_modules=[HOG_module])
