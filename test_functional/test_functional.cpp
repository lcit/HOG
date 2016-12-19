/*  =========================================================================
    Author: Leonardo Citraro
    Company:
    Filename: test_functional.cpp
    Last modifed:   19.12.2016 by Leonardo Citraro
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

int main(int argc, char* argv[]) {

    try {
        HOG hog(16, 8, 8, 9, HOG::GRADIENT_UNSIGNED, HOG::L2hys, 1);
    } catch(...) {
        std::cout << "Test correct constructor failed!\n"; throw;
    }
    
    try {
        HOG hog(16, 8, 8, 1, HOG::GRADIENT_UNSIGNED, HOG::L2hys, 1);
        std::cout << "Test binning=1 failed!\n";  exit(-1);
    } catch(...) { }
    try {
        HOG hog(16, 8, 7, 2, HOG::GRADIENT_UNSIGNED, HOG::L2hys, 1);
        std::cout << "Test stride<cellsize failed!\n";  exit(-1);
    } catch(...) { }
    try {
        HOG hog(12, 6, 8, 1, HOG::GRADIENT_UNSIGNED, HOG::L2hys, 1);
        std::cout << "Test stride not a multiple of cellsize failed!\n";  exit(-1);
    } catch(...) { }
    try {
        HOG hog(15, 8, 8, 1, HOG::GRADIENT_UNSIGNED, HOG::L2hys, 1);
        std::cout << "Test blocksize not a multiple of cellsize failed!\n";  exit(-1);
    } catch(...) { }
    try {
        HOG hog(16, 8, 8, 1, 4, HOG::L2hys, 1);
        std::cout << "Test grad_type unknown failed!\n";  exit(-1);
    } catch(...) { }

    HOG hog(16, 8, 8, 9, HOG::GRADIENT_UNSIGNED, HOG::L2hys, 1);
    try {
        hog.process(cv::Mat::ones(300,200,CV_8U));
    } catch(...) {
        std::cout << "Test correct process failed!\n"; exit(-1);
    }
    try {
        hog.process(cv::Mat::ones(16,16,CV_8U));
    } catch(...) {
        std::cout << "Test correct process 2 failed!\n"; exit(-1);
    }
    try {
        hog.process(cv::Mat::ones(15,16,CV_8U));
        std::cout << "Test image size < blocksize failed!\n"; exit(-1);
    } catch(...) { }
    try {
        hog.process(cv::Mat::ones(16,15,CV_8U));
        std::cout << "Test image size < blocksize failed!\n"; exit(-1);
    } catch(...) { }
    
    
    try {
        hog.process(cv::Mat::ones(16,16,CV_8U));
        
        auto hist = hog.retrieve(cv::Rect(0,0,16,16));
        if(hist.size() != 36) {
            std::cout << "Test correct retrieve failed (hist size wrong)! " << hist.size() << "\n"; exit(-1);
        }
    } catch(...) { 
        std::cout << "Test correct retrieve failed!\n"; throw;
    }
    try {
        hog.process(cv::Mat::ones(32,16,CV_8U));
        auto hist = hog.retrieve(cv::Rect(0,0,16,32));
        if(hist.size() != 108) {
            std::cout << "Test correct retrieve 2 failed (hist size wrong)! " << hist.size() << "\n"; exit(-1);
        }
    } catch(...) { 
        std::cout << "Test correct retrieve 2 failed!\n"; throw;
    }
    try {
        hog.process(cv::Mat::ones(32,16,CV_8U));
        auto hist = hog.retrieve(cv::Rect(0,0,16,33));
        std::cout << "Test rect.width > img.width failed!\n"; exit(-1);
    } catch(...) { }
    try {
        hog.process(cv::Mat::ones(32,16,CV_8U));
        auto hist = hog.retrieve(cv::Rect(0,0,17,32));
        std::cout << "Test rect.height > img.height failed!\n"; exit(-1);
    } catch(...) { }
    try {
        hog.process(cv::Mat::ones(32,16,CV_8U));
        auto hist = hog.retrieve(cv::Rect(1,0,16,32));
        std::cout << "Test window.x out of image failed!\n"; exit(-1);
    } catch(...) { }
    try {
        hog.process(cv::Mat::ones(32,16,CV_8U));
        auto hist = hog.retrieve(cv::Rect(0,1,16,32));
        std::cout << "Test window.y out of image failed!\n"; exit(-1);
    } catch(...) { }
    

    std::cout << "\nTest passed!\n\n"; exit(0);
    return 0;

}
