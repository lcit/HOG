/*  =========================================================================
    Author: Leonardo Citraro
    Company:
    Filename: HOG.cpp
    Last modifed:   12.12.2016 by Leonardo Citraro
    Description:    Straightforward (CPU based) implementation of the
                    HOG (Histogram of Oriented Gradients) using OpenCV

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
#include <vector>
#include <functional>
#include <math.h>

// see: https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients#Block_normalization
void HOG::L1norm(HOG::THist& v) {
    HOG::TType den = std::accumulate(std::begin(v), std::end(v), 0.0f) + epsilon;

    if (den != 0)
        std::transform(std::begin(v), std::end(v), std::begin(v), [den](const HOG::TType nom) {
        return nom / den;
    });
}

void HOG::L1sqrt(HOG::THist& v) {
    HOG::L1norm(v);
    std::transform(std::begin(v), std::end(v), std::begin(v), [](const HOG::TType x) {
        return std::sqrt(x);
    });
}

void HOG::L2norm(HOG::THist& v) {
    HOG::THist temp = v;
    std::transform(std::begin(v), std::end(v), std::begin(temp), [](const HOG::TType & x) {
        return x * x;
    });
    HOG::TType den = std::accumulate(std::begin(temp), std::end(temp), 0.0f);
    den = std::sqrt(den + epsilon);

    if (den != 0)
        std::transform(std::begin(v), std::end(v), std::begin(v), [den](const HOG::TType nom) {
        return nom / den;
    });
}

void HOG::L2hys(HOG::THist& v) {
    HOG::L2norm(v);
    auto clip = [](const HOG::TType & x) {
        if (x > 0.2) return 0.2f;
        else if (x < 0) return 0.0f;
        else return x;
    };
    std::transform(std::begin(v), std::end(v), std::begin(v), clip);
    HOG::L2norm(v);
}

void HOG::none(HOG::THist& v) {}

HOG::HOG(const size_t blocksize, std::function<void(HOG::THist&)> block_norm, const unsigned n_threads)
    : _blocksize(blocksize), _cellsize(blocksize / 2), _stride(blocksize / 2),
      _binning(9), _grad_type(GRADIENT_UNSIGNED), _bin_width(_grad_type / _binning), 
      _block_norm(block_norm), _n_threads(n_threads) {}
HOG::HOG(const size_t blocksize, size_t cellsize,
         std::function<void(HOG::THist&)> block_norm, const unsigned n_threads)
    : _blocksize(blocksize), _cellsize(cellsize), _stride(blocksize / 2), _binning(9),
      _grad_type(GRADIENT_UNSIGNED), _bin_width(_grad_type / _binning), _block_norm(block_norm), _n_threads(n_threads) {}
HOG::HOG(const size_t blocksize, size_t cellsize, size_t stride,
         std::function<void(HOG::THist&)> block_norm, const unsigned n_threads)
    : _blocksize(blocksize), _cellsize(cellsize), _stride(stride), _binning(9),
      _grad_type(GRADIENT_UNSIGNED), _bin_width(_grad_type / _binning), _block_norm(block_norm), _n_threads(n_threads) {}
HOG::HOG(const size_t blocksize, size_t cellsize, size_t stride, size_t binning, size_t grad_type,
         std::function<void(HOG::THist&)> block_norm, const unsigned n_threads)
    : _blocksize(blocksize), _cellsize(cellsize), _stride(stride), _binning(binning),
      _grad_type(grad_type), _bin_width(_grad_type / _binning), _block_norm(block_norm), _n_threads(n_threads) {}
HOG::~HOG() {}

void HOG::process(const cv::Mat& img) {
    
    // cleanup
    clear_internals();
    
    // makes sure the image is normalized
    cv::normalize(img, norm, 0.0, 255.0, cv::NORM_MINMAX, CV_32F);

    // extracts the magnitude and orientations images
    magnitude_and_orientation(img);
    
    // iterates over all blocks and cells
    //#pragma omp parallel num_threads(_n_threads)
    {
        _n_cells_y = static_cast<int>(mag.rows/_cellsize);
        _n_cells_x = static_cast<int>(mag.cols/_cellsize);
        
        //std::cout << "_n_cells_y=" << _n_cells_y << ", _n_cells_x=" << _n_cells_x << "\n";
        
        _cell_hists.resize(_n_cells_y);
        
        // There might be a bug in openmp, we can't safely fill an std::vector with fixed size
        // http://stackoverflow.com/questions/30815669/error-with-openmp-for-nested-for-loop
        //#pragma omp for collapse(2) 
        for (size_t i = 0; i < _n_cells_y; ++i) {
            _cell_hists[i].resize(_n_cells_x);
            for (size_t j = 0; j < _n_cells_x; ++j) {
                cv::Rect cell_rect = cv::Rect(j*_cellsize, i*_cellsize, _cellsize, _cellsize);
                HOG::THist cell_hist = process_cell(cv::Mat(mag, cell_rect), cv::Mat(ori, cell_rect));
                _cell_hists[i][j] = cell_hist;
            }
        }
    }
}

HOG::THist HOG::retrieve(const cv::Rect& rect) {
    size_t x = static_cast<int>(rect.x/_cellsize);
    size_t y = static_cast<int>(rect.y/_cellsize);
    size_t width = static_cast<int>(rect.width/_cellsize);
    size_t height = static_cast<int>(rect.height/_cellsize);
    /*
    static_assert(y<_cell_hists.size(), "Error: Rect.y is greater than the image!");
    static_assert(x<_cell_hists[0].size(), "Error: Rect.x is greater than the image!");
    static_assert(width>=_n_cells_per_block_x, "Error: Rect.width is smaller than blocksize!");
    static_assert(height>=_n_cells_per_block_y, "Error: Rect.height is smaller than blocksize!");
    */
    HOG::THist hog_hist;
    for(size_t block_y=y; block_y<y+height-_n_cells_per_block_y; block_y += _stride_unit) {
        for(size_t block_x=x; block_x<x+width-_n_cells_per_block_x; block_x += _stride_unit) {
            HOG::THist block_hist;
            //block_hist.resize(_n_cells_per_block*_binning);
            for(size_t cell_y=block_y; cell_y<block_y+_n_cells_per_block_y; ++cell_y) {
                for(size_t cell_x=block_x; cell_x<block_x+_n_cells_per_block_x; ++cell_x) {
                    
                    //std::cout << "block_y=" << block_y << ", block_x=" << block_x << ", cell_y=" << cell_y << ", cell_x=" << cell_x << "\n";
                    
                    THist cell_hist = _cell_hists[cell_y][cell_x];
                    block_hist.insert(std::end(block_hist), std::begin(cell_hist), std::end(cell_hist));
                }
            }
            _block_norm(block_hist);
            hog_hist.insert(std::end(hog_hist), std::begin(block_hist), std::end(block_hist));
        }
    }
    return hog_hist;
}

void HOG::magnitude_and_orientation(const cv::Mat& img) {
    cv::Mat Dx, Dy;
    cv::filter2D(img, Dx, CV_32F, _kernelx);
    cv::filter2D(img, Dy, CV_32F, _kernely);
    cv::magnitude(Dx, Dy, mag);
    cv::phase(Dx, Dy, ori, true);
}

HOG::THist HOG::process_cell(const cv::Mat& cell_mag, const cv::Mat& cell_ori) {
    HOG::THist cell_hist(_binning);
    #pragma omp parallel num_threads(_n_threads)
    {
        if(_grad_type == GRADIENT_SIGNED) {
            #pragma omp for
            for (size_t i = 0; i < cell_mag.rows; ++i) {
                const HOG::TType* ptr_row_mag = cell_mag.ptr<HOG::TType>(i);
                const HOG::TType* ptr_row_ori = cell_ori.ptr<HOG::TType>(i);
                for (size_t j = 0; j < cell_mag.cols; ++j) {
                    cell_hist[static_cast<int>(ptr_row_ori[j] / _bin_width)] += ptr_row_mag[j];
                }
            }
        } else {
            #pragma omp for
            for (size_t i = 0; i < cell_mag.rows; ++i) {
                const HOG::TType* ptr_row_mag = cell_mag.ptr<HOG::TType>(i);
                const HOG::TType* ptr_row_ori = cell_ori.ptr<HOG::TType>(i);
                for (size_t j = 0; j < cell_mag.cols; ++j) {
                    HOG::TType orientation = ptr_row_ori[j];
                    if(orientation > 180)
                        orientation -= 180;
                    cell_hist[static_cast<int>(orientation / _bin_width)] += ptr_row_mag[j];
                }
            }
        }
    }
    return cell_hist;
}

cv::Mat HOG::get_magnitudes() {
    return mag;
}

cv::Mat HOG::get_orientations() {
    return ori;
}

cv::Mat HOG::get_vector_mask() {
    cv::Mat vector_mask = cv::Mat::zeros(norm.size(), CV_8U);
    
    float max = 0;
    
    std::vector<std::vector<float>> cell_hist_maxs;
    cell_hist_maxs.resize(_n_cells_y);
    for (size_t i = 0; i < _n_cells_y; ++i) {
        cell_hist_maxs[i].resize(_n_cells_x);
        for (size_t j = 0; j < _n_cells_x; ++j) {
            HOG::THist cell_hist = _cell_hists[i][j];
            HOG::TType cell_hist_max = *std::max_element(std::begin(cell_hist), std::end(cell_hist));
            cell_hist_maxs[i][j] = cell_hist_max;
            if(cell_hist_max > max)
                max = cell_hist_max;
        }
    }
    
    for (size_t i = 0; i < _n_cells_y; ++i) {
        for (size_t j = 0; j < _n_cells_x; ++j) {
            HOG::THist cell_hist = _cell_hists[i][j];

            int color_magnitude = static_cast<int>(cell_hist_maxs[i][j] / max * 255.0);
            //std::cout << "color_magnitude=" << color_magnitude << "\n";

            // iterates over the cell histogram
            for (size_t k = 0; k < cell_hist.size(); ++k) {
                // fixed line thinkness
                int thickness = 1;

                // length of the "arrows"
                int length = static_cast<int>((cell_hist[k] / cell_hist_maxs[i][j]) * _cellsize / 2);

                if (length > 0 && !isinf(length)) {
                    // draw "arrows" of varing length
                    if(_grad_type == GRADIENT_SIGNED) {
                        cv::line(vector_mask, cv::Point(j*_cellsize + _cellsize / 2, i*_cellsize + _cellsize / 2),
                             cv::Point(  j*_cellsize + _cellsize / 2 + cos((k * _bin_width) * 3.1415 / 180)*length,
                                         i*_cellsize + _cellsize / 2 + sin((k * _bin_width) * 3.1415 / 180)*length),
                             cv::Scalar(color_magnitude, color_magnitude, color_magnitude), thickness);
                    } else {
                        cv::line(vector_mask, 
                            cv::Point(  j*_cellsize + _cellsize / 2 + cos((k * _bin_width+180) * 3.1415 / 180)*length,
                                         i*_cellsize + _cellsize / 2 + sin((k * _bin_width+180) * 3.1415 / 180)*length),
                             cv::Point(  j*_cellsize + _cellsize / 2 + cos((k * _bin_width) * 3.1415 / 180)*length,
                                         i*_cellsize + _cellsize / 2 + sin((k * _bin_width) * 3.1415 / 180)*length),
                             cv::Scalar(color_magnitude, color_magnitude, color_magnitude), thickness);
                    }
                }
            }
            // draw cell delimiters
            cv::line(vector_mask, cv::Point(j*_cellsize, i*_cellsize), cv::Point(j*_cellsize + norm.rows, i*_cellsize), cv::Scalar(255, 255, 255), 1);
            cv::line(vector_mask, cv::Point(j*_cellsize, i*_cellsize), cv::Point(j*_cellsize, i*_cellsize + norm.rows), cv::Scalar(255, 255, 255), 1);
        }
    }

    return vector_mask;
}

void HOG::clear_internals() {
    img_hist.clear();
    for(auto& h:_all_hists) 
        h.clear();
    _all_hists.clear();
    for(auto& h1:_cell_hists) {
        for(auto& h2:h1) 
            h2.clear();
        h1.clear();
    }
    _cell_hists.clear();
}
