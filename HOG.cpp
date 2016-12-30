/*  ==========================================================================================
    Author: Leonardo Citraro
    Company:
    Filename: HOG.cpp
    Last modifed:   28.12.2016 by Leonardo Citraro
    Description:    Straightforward (CPU based) implementation of the
                    HOG (Histogram of Oriented Gradients) using OpenCV.
                    https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf

    ==========================================================================================
    Copyright (c) 2016 Leonardo Citraro <ldo.citraro@gmail.com>

    Permission is hereby granted, free of charge, to any person obtaining a copy of this
    software and associated documentation files (the "Software"), to deal in the Software
    without restriction, including without limitation the rights to use, copy, modify,
    merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following
    conditions:

    The above copyright notice and this permission notice shall be included in all copies
    or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
    ==========================================================================================
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
#include <iomanip>
#include <fstream>

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

std::function<void(HOG::THist&)> get_block_norm(const HOG::BLOCK_NORM norm) {
    if(norm == HOG::BLOCK_NORM::none)
        return HOG::none;
    else if (norm == HOG::BLOCK_NORM::L1norm)
        return HOG::L1norm;
    else if (norm == HOG::BLOCK_NORM::L1sqrt)
        return HOG::L1sqrt;
    else if (norm == HOG::BLOCK_NORM::L2norm)
        return HOG::L2norm;
    else if (norm == HOG::BLOCK_NORM::L2hys)
        return HOG::L2hys;
    else
        return HOG::none;
}

void check_ctor_params(const size_t blocksize, const size_t cellsize, const size_t stride, 
                        const size_t binning, const size_t grad_type) {
    if(blocksize < 2)
        throw std::runtime_error("HOG::HOG(): blocksize must be at least 2 pixels!");
    if(cellsize < 1)
        throw std::runtime_error("HOG::HOG(): cellsize must be at least 1 pixels!");
    if(binning < 2)
        throw std::runtime_error("HOG::HOG(): binning should at least be greater or equal to 2!");
    if(grad_type != HOG::GRADIENT_UNSIGNED && grad_type != HOG::GRADIENT_SIGNED)
        throw std::runtime_error("HOG::HOG(): grad_type entered doesn't match the default identifiers!");
    if(blocksize%cellsize != 0)
        throw std::runtime_error("HOG::HOG(): blocksize must be a multiple of cellsize!");
    if(stride%cellsize != 0)
        throw std::runtime_error("HOG::HOG(): stride must be a multiple of cellsize!");
}

HOG::HOG()
    : _blocksize(16), _cellsize(8), _stride(8), _binning(9), _grad_type(GRADIENT_UNSIGNED), 
      _bin_width(_grad_type / _binning), _block_norm(HOG::none), _norm_function(HOG::BLOCK_NORM::none){
        check_ctor_params(_blocksize, _cellsize, _stride, _binning, _grad_type);
    }
HOG::HOG(const size_t blocksize, const HOG::BLOCK_NORM block_norm)
    : _blocksize(blocksize), _cellsize(blocksize / 2), _stride(blocksize / 2),
      _binning(9), _grad_type(GRADIENT_UNSIGNED), _bin_width(_grad_type / _binning), 
      _block_norm(get_block_norm(block_norm)), _norm_function(block_norm) {
        check_ctor_params(_blocksize, _cellsize, _stride, _binning, _grad_type);
    }
HOG::HOG(const size_t blocksize, const size_t cellsize,
         const HOG::BLOCK_NORM block_norm)
    : _blocksize(blocksize), _cellsize(cellsize), _stride(blocksize / 2), _binning(9),
      _grad_type(GRADIENT_UNSIGNED), _bin_width(_grad_type / _binning), _block_norm(get_block_norm(block_norm)),
      _norm_function(block_norm) {
        check_ctor_params(_blocksize, _cellsize, _stride, _binning, _grad_type);
    }
HOG::HOG(const size_t blocksize, const size_t cellsize, const size_t stride,
         const HOG::BLOCK_NORM block_norm)
    : _blocksize(blocksize), _cellsize(cellsize), _stride(stride), _binning(9),
      _grad_type(GRADIENT_UNSIGNED), _bin_width(_grad_type / _binning), _block_norm(get_block_norm(block_norm)),
      _norm_function(block_norm) {
        check_ctor_params(_blocksize, _cellsize, _stride, _binning, _grad_type);
    }
HOG::HOG(const size_t blocksize, const size_t cellsize, const size_t stride, 
        const size_t binning, const size_t grad_type, const HOG::BLOCK_NORM block_norm)
    : _blocksize(blocksize), _cellsize(cellsize), _stride(stride), _binning(binning),
      _grad_type(grad_type), _bin_width(_grad_type / _binning), _block_norm(get_block_norm(block_norm)), 
      _norm_function(block_norm) {
        check_ctor_params(_blocksize, _cellsize, _stride, _binning, _grad_type);
    }
HOG::~HOG() {}

// Copy constructor
HOG::HOG(const HOG& to_copy) 
    : _blocksize(to_copy._blocksize), _cellsize(to_copy._cellsize), _stride(to_copy._stride), _binning(to_copy._binning),
      _grad_type(to_copy._grad_type), _bin_width(_grad_type / _binning), _block_norm(to_copy._block_norm),
      _norm_function(to_copy._norm_function) {
    }
    
// assignment operator
HOG& HOG::operator=(const HOG& to_copy) {
    _blocksize = to_copy._blocksize;
    _cellsize = to_copy._cellsize;
    _stride = to_copy._stride;
    _binning = to_copy._binning;
    _grad_type = to_copy._grad_type;
    _bin_width = to_copy._bin_width;
    _norm_function = to_copy._norm_function;
    _block_norm = to_copy._block_norm;
    _n_cells_per_block_y = _blocksize/_cellsize;
    _n_cells_per_block_x = _n_cells_per_block_y;
    _n_cells_per_block = _n_cells_per_block_y*_n_cells_per_block_x;
    _block_hist_size = _binning*_n_cells_per_block;
    _stride_unit = _stride/_cellsize;
    return *this;
}

void HOG::process(const cv::Mat& img) {
    
    if(!img.data)
        throw std::runtime_error("HOG::process(): invalid image!");
    if(img.rows < _blocksize || img.cols < _blocksize)
        throw std::runtime_error("HOG::process(): the image is smaller than blocksize!");
    
    // cleanup
    clear_internals();

    // extracts the magnitude and orientations images
    magnitude_and_orientation(img);
    
    _n_cells_y = static_cast<int>(mag.rows/_cellsize);
    _n_cells_x = static_cast<int>(mag.cols/_cellsize);
    
    _cell_hists.resize(_n_cells_y);
    
    // iterates over all blocks and cells
    // We tried to use OpenMP here but with scarce results. The function process_cell()
    // doesn't consume a great deal of CPU so OpenMP struggle to spread the computation
    // over multiple threads. The real time-consuming block of code here is the function retrieve().
    for (size_t i = 0; i < _n_cells_y; ++i) {
        _cell_hists[i].resize(_n_cells_x);
        for (size_t j = 0; j < _n_cells_x; ++j) {
            cv::Rect cell_rect = cv::Rect(j*_cellsize, i*_cellsize, _cellsize, _cellsize);
            const HOG::THist cell_hist = process_cell(cv::Mat(mag, cell_rect), cv::Mat(ori, cell_rect));
            _cell_hists[i][j] = cell_hist;
        }
        
    }
}

const HOG::THist HOG::retrieve(const cv::Rect& window) {
    
    if(window.height < _blocksize || window.width < _blocksize)
        throw std::runtime_error("HOG::retrieve(): the window is smaller than blocksize!");
    if(window.x > mag.cols-window.width || window.y > mag.rows-window.height)
        throw std::runtime_error("HOG::retrieve(): the window goes outside of the bounds of the image!");
    
    // convert the window pixels into cell-units so we can iterate over 
    // the vector of vectors of cell histograms (_cell_hists)
    size_t x = static_cast<int>(window.x/_cellsize);
    size_t y = static_cast<int>(window.y/_cellsize);
    size_t width = static_cast<int>(window.width/_cellsize);
    size_t height = static_cast<int>(window.height/_cellsize);
    
    // Also here we tried to use OpenMP but with scarce results.
    HOG::THist hog_hist;
    for(size_t block_y=y; block_y<=y+height-_n_cells_per_block_y; block_y += _stride_unit) {
        for(size_t block_x=x; block_x<=x+width-_n_cells_per_block_x; block_x += _stride_unit) {
            HOG::THist block_hist;
            block_hist.reserve(_block_hist_size);
            for(size_t cell_y=block_y; cell_y<block_y+_n_cells_per_block_y; ++cell_y) {
                for(size_t cell_x=block_x; cell_x<block_x+_n_cells_per_block_x; ++cell_x) {
                    const THist cell_hist = _cell_hists[cell_y][cell_x];
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

const HOG::THist HOG::process_cell(const cv::Mat& cell_mag, const cv::Mat& cell_ori) {
    HOG::THist cell_hist(_binning);
    if(_grad_type == GRADIENT_SIGNED) {
        for (size_t i = 0; i < cell_mag.rows; ++i) {
            const HOG::TType* ptr_row_mag = cell_mag.ptr<HOG::TType>(i);
            const HOG::TType* ptr_row_ori = cell_ori.ptr<HOG::TType>(i);
            for (size_t j = 0; j < cell_mag.cols; ++j) {
                cell_hist.at(static_cast<int>(ptr_row_ori[j] / _bin_width)) += ptr_row_mag[j];
            }
        }
    } else {
        for (size_t i = 0; i < cell_mag.rows; ++i) {
            const HOG::TType* ptr_row_mag = cell_mag.ptr<HOG::TType>(i);
            const HOG::TType* ptr_row_ori = cell_ori.ptr<HOG::TType>(i);
            for (size_t j = 0; j < cell_mag.cols; ++j) {
                HOG::TType orientation = ptr_row_ori[j];
                if(orientation >= 180)
                    orientation -= 180;
                cell_hist.at(static_cast<int>(orientation / _bin_width)) += ptr_row_mag[j];
            }
        }
    }
    return cell_hist;
}

const cv::Mat HOG::get_magnitudes() {
    return mag;
}

const cv::Mat HOG::get_orientations() {
    return ori;
}

const cv::Mat HOG::get_vector_mask(const int thickness) {
    cv::Mat vector_mask = cv::Mat::zeros(mag.size(), CV_8U);
    
    // the maximum value of all cell histogram of the image
    float max = 0;

    // iterate through all cells in the image to get the local hist max value 
    // and the max value of the entire image
    std::vector<std::vector<float>> cell_hist_maxs(_n_cells_y);
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
    
    // iterate through all cells in the image
    for (size_t i = 0; i < _n_cells_y; ++i) {
        for (size_t j = 0; j < _n_cells_x; ++j) {
            HOG::THist cell_hist = _cell_hists[i][j];

            // the color of the lines depends uppon the local hist max and the overall max
            int color_magnitude = static_cast<int>(cell_hist_maxs[i][j] / max * 255.0);

            // iterates over the cell histogram
            for (size_t k = 0; k < cell_hist.size(); ++k) {

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
            cv::line(vector_mask, cv::Point(j*_cellsize-1, i*_cellsize-1), cv::Point(j*_cellsize + mag.rows-1, i*_cellsize-1), cv::Scalar(255, 255, 255), thickness);
            cv::line(vector_mask, cv::Point(j*_cellsize-1, i*_cellsize-1), cv::Point(j*_cellsize-1, i*_cellsize + mag.rows-1), cv::Scalar(255, 255, 255), thickness);
        }
    }

    return vector_mask;
}

void HOG::clear_internals() {
    for(auto& h1:_cell_hists) {
        for(auto& h2:h1) 
            h2.clear();
        h1.clear();
    }
    _cell_hists.clear();
}

void HOG::save(const std::string& filename) {
    try {
        std::ofstream f(filename, std::ios::binary);
        f.write( (char*)&_blocksize, sizeof(_blocksize) );
        f.write( (char*)&_cellsize, sizeof(_cellsize) );
        f.write( (char*)&_stride, sizeof(_stride) );
        f.write( (char*)&_binning, sizeof(_binning) );
        f.write( (char*)&_grad_type, sizeof(_grad_type) );
        f.write( (char*)&_bin_width, sizeof(_bin_width) );
        f.write( (char*)&_norm_function, sizeof(_norm_function) );
        f.close();
    } catch(...) {
        throw;
    }
}

HOG HOG::load(const std::string& filename) {
    
    size_t blocksize;
    size_t cellsize;
    size_t stride;
    size_t grad_type;
    size_t binning;
    BLOCK_NORM norm_function;
    
    try {
        std::ifstream f(filename, std::ios::binary);
        f.read( (char*)&blocksize, sizeof(blocksize) );
        f.read( (char*)&cellsize, sizeof(cellsize) );
        f.read( (char*)&stride, sizeof(stride) );
        f.read( (char*)&binning, sizeof(binning) );
        f.read( (char*)&grad_type, sizeof(grad_type) );
        f.read( (char*)&norm_function, sizeof(norm_function) );
        f.close();
        
        return HOG(blocksize, cellsize, stride, binning, grad_type, norm_function);
    } catch(...) {
        throw;
    }
}
