/*  =========================================================================
    Author: Leonardo Citraro
    Company:
    Filename: main.cpp
    Last modifed:   11.12.2016 by Leonardo Citraro
    Description:    Test of the HOG feature

    =========================================================================
    https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
    =========================================================================
*/
#include "HOG.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <algorithm>
#include <memory>
#include <functional>
#include <math.h>

void display_superimposed(const cv::Mat& A, const cv::Mat& B, const std::string& name) {
    cv::Mat superimposed;
    cv::addWeighted(A, 0.5, B, 0.5, 0.0, superimposed);
    imshow(name, superimposed);
}

cv::Mat custom_normalization(const cv::Mat& src) {
    double min, max;
    cv::minMaxLoc(src, &min, &max);
    cv::Mat dst = src * 200 / (max - min) + 128;
    dst.convertTo(dst, CV_8U);
    return dst;
}

int main(int argc, char* argv[]) {

    std::vector<float> v = {1, 6, 7, 2, 2, 2, 3};
    HOG::L2norm(v);

    for (auto vv : v)
        std::cout << vv << ",";

    // open and display an image
    cv::Mat image = cv::imread(argv[1], CV_8U);
    cv::imshow("original", image);

    // Retrieve the HOG from the image
    size_t blocksize = atoi(argv[2]);
    size_t cellsize = atoi(argv[3]);
    size_t stride = atoi(argv[4]);
    size_t binning = atoi(argv[5]);
    HOG hog(blocksize, cellsize, stride, binning, HOG::GRADIENT_UNSIGNED);
    hog.process(image);
    auto hist = hog.retrieve(cv::Rect(0,0,image.cols, image.rows));
    
    // print the resulting histogram
    std::cout << "Histogram size: " << hist.size() << "\n";

    for (auto h : hist)
        std::cout << h << ",";

    std::cout << "\n";
    /*
    // display some usefull images
    display_superimposed(image, hog.get_vector_mask(), "vector_mask");
    display_superimposed(custom_normalization(hog.get_magnitudes()), hog.get_vector_mask(), "magnitude");
    display_superimposed(custom_normalization(hog.get_orientations()), hog.get_vector_mask(), "orientation");
    */
    
    
    cv::waitKey();

    return 0;

}
