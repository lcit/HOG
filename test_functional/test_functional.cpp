/*  =========================================================================
    Author: Leonardo Citraro
    Company:
    Filename: test_functional.cpp
    Last modifed:   29.12.2016 by Leonardo Citraro
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
#include <iomanip>

int main(int argc, char* argv[]) {

    try {
        HOG hog(16, 8, 8, 9, HOG::GRADIENT_UNSIGNED, HOG::L2hys);
    } catch(...) {
        std::cout << "Test correct constructor failed!\n"; throw;
    }
    try {
        HOG hog(16, 8, 8, 1, HOG::GRADIENT_UNSIGNED, HOG::L2hys);
        std::cout << "Test binning=1 failed!\n";  exit(-1);
    } catch(...) { }
    try {
        HOG hog(16, 8, 7, 2, HOG::GRADIENT_UNSIGNED, HOG::L2hys);
        std::cout << "Test stride<cellsize failed!\n";  exit(-1);
    } catch(...) { }
    try {
        HOG hog(12, 6, 8, 1, HOG::GRADIENT_UNSIGNED, HOG::L2hys);
        std::cout << "Test stride not a multiple of cellsize failed!\n";  exit(-1);
    } catch(...) { }
    try {
        HOG hog(15, 8, 8, 1, HOG::GRADIENT_UNSIGNED, HOG::L2hys);
        std::cout << "Test blocksize not a multiple of cellsize failed!\n";  exit(-1);
    } catch(...) { }
    try {
        HOG hog(16, 8, 8, 1, 4, HOG::L2hys);
        std::cout << "Test grad_type unknown failed!\n";  exit(-1);
    } catch(...) { }

    HOG hog(16, 8, 8, 9, HOG::GRADIENT_UNSIGNED, HOG::L2hys);
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

    {
        cv::Mat img = cv::Mat::ones(32,32,CV_8U);
        hog.process(img);
        auto hist = hog.retrieve(cv::Rect(0,0,32,32));
        const cv::Mat mag = hog.get_magnitudes();
        if(mag.size() != img.size()) {
            std::cout << "Test mag.size() != img.size() failed! " << mag.size() << "\n"; 
            exit(-1);
        }
        const cv::Mat ori = hog.get_orientations();
        if(ori.size() != img.size()) {
            std::cout << "Test ori.size() != img.size() failed! " << ori.size() << "\n"; 
            exit(-1);
        }
        const cv::Mat vector_mask = hog.get_vector_mask(1);
        if(vector_mask.size() != img.size()) {
            std::cout << "Test vector_mask.size() != img.size() failed! " << vector_mask.size() << "\n"; 
            exit(-1);
        }
    }
    
    {   // This test verifies if the HOG histograms are the same
        // in the case when we convert an image that has already the size 
        // of the window and an image that as the same size but that it 
        // have been extracted form a bigger image using HOG::retrieve().
        
        // full image
        cv::Mat image = cv::imread("../img/astronaut.JPG", CV_8U);
        
        // window image
        cv::Rect r = cv::Rect(0,0,64,64);
        cv::Mat sub = cv::Mat(image, r);
        
        HOG hog(64, 32, 32, 9, HOG::GRADIENT_SIGNED, HOG::none);
        
        hog.process(sub);
        auto hist1 = hog.retrieve(cv::Rect(0,0,r.width,r.height));
        
        hog.process(image);
        auto hist2 = hog.retrieve(r);
        
        // hist1 and hist2 should give the same result
        for(int i=0; i<hist1.size(); ++i) {
            if((hist1[i]-hist2[i])>10) {
                std::cout << "Test full image vs. failed!\n";  exit(-1);
            }
        }
    }
    
    std::cout << "\nTest passed!\n\n"; exit(0);
    
    return 0;
}
