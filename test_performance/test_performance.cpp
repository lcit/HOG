/*  =========================================================================
    Author: Leonardo Citraro
    Company:
    Filename: test_performance.cpp
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
#include <functional>
#include <chrono>

template<typename TimeT = std::chrono::milliseconds>
struct measure
{
    template<typename F, typename ...Args>
    static typename TimeT::rep run(F&& func, Args&&... args)
    {
        auto start = std::chrono::steady_clock::now();
        std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto duration = std::chrono::duration_cast<TimeT>(std::chrono::steady_clock::now() - start);
        return duration.count();
    }
};

template<size_t N>
struct mean_stddev {
    template<typename F, typename ...Args>
    static auto run(F&& func, Args&&... args){
        std::array<double, N> buffer;
        for(auto& buf : buffer)
            buf = std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
        auto sum = std::accumulate(std::begin(buffer), std::end(buffer), 0.0);
        auto mean = sum/buffer.size();
        std::array<double, N> diff;
        std::transform(std::begin(buffer), std::end(buffer), std::begin(diff), [mean](auto x) { return x - mean; });
        auto sq_sum = std::inner_product(std::begin(diff), std::end(diff), std::begin(diff), 0.0);
        auto stddev = std::sqrt(sq_sum/buffer.size());
        return std::make_pair(mean,stddev);
    }
};

int main(int argc, char* argv[]) {

    auto function = [](const int n_threads){
        cv::Mat image = cv::imread("00001665.jpg", CV_8U);

        // Retrieve the HOG from the image
        size_t cellsize = 8;
        size_t blocksize = cellsize*2;
        size_t stride = cellsize;
        size_t binning = 9;
        HOG hog(blocksize, cellsize, stride, binning, HOG::GRADIENT_UNSIGNED, HOG::L2hys, n_threads);
        hog.process(image);
        
        cv::Size window(50,100);
        
        for(int x=0; x<image.cols-window.width; x += cellsize){
             for(int y=0; y<image.rows-window.height; y += cellsize){
                cv::Rect r = cv::Rect(x,y, window.width, window.height);
                auto hist = hog.retrieve(r);
            }
        }
    };
    
    auto res = mean_stddev<3>::run([&](){return measure<>::run(function,1);});
    std::cout << "Time elapsed (n_threads=1): " << res.first << "(+-" << res.second << ") [ms]\n";
    res = mean_stddev<3>::run([&](){return measure<>::run(function,2);});
    std::cout << "Time elapsed (n_threads=2): " << res.first << "(+-" << res.second << ") [ms]\n";
    res = mean_stddev<3>::run([&](){return measure<>::run(function,4);});
    std::cout << "Time elapsed (n_threads=4): " << res.first << "(+-" << res.second << ") [ms]\n";
    res = mean_stddev<3>::run([&](){return measure<>::run(function,8);});
    std::cout << "Time elapsed (n_threads=8): " << res.first << "(+-" << res.second << ") [ms]\n";

    return 0;

}

/*
    Time elapsed (n_threads=1): 124.333(+-4.02768) [ms]
    Time elapsed (n_threads=2): 150.333(+-0.942809) [ms]
    Time elapsed (n_threads=4): 148.333(+-1.24722) [ms]
    Time elapsed (n_threads=8): 264.667(+-77.933) [ms]
*/
