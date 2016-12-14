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

HOG::HOG(const size_t blocksize, std::function<void(HOG::THist&)> block_norm)
    : _blocksize(blocksize), _cellsize(blocksize / 2), _stride(blocksize / 2),
      _binning(9), _grad_type(GRADIENT_UNSIGNED), _bin_width(_grad_type / _binning), _block_norm(block_norm) {}
HOG::HOG(const size_t blocksize, size_t cellsize,
         std::function<void(HOG::THist&)> block_norm)
    : _blocksize(blocksize), _cellsize(cellsize), _stride(blocksize / 2), _binning(9),
      _grad_type(GRADIENT_UNSIGNED), _bin_width(_grad_type / _binning), _block_norm(block_norm) {}
HOG::HOG(const size_t blocksize, size_t cellsize, size_t stride,
         std::function<void(HOG::THist&)> block_norm)
    : _blocksize(blocksize), _cellsize(cellsize), _stride(stride), _binning(9),
      _grad_type(GRADIENT_UNSIGNED), _bin_width(_grad_type / _binning), _block_norm(block_norm) {}
HOG::HOG(const size_t blocksize, size_t cellsize, size_t stride, size_t binning, size_t grad_type,
         std::function<void(HOG::THist&)> block_norm)
    : _blocksize(blocksize), _cellsize(cellsize), _stride(stride), _binning(binning),
      _grad_type(grad_type), _bin_width(_grad_type / _binning), _block_norm(block_norm) {}
HOG::~HOG() {}

HOG::THist HOG::convert(const cv::Mat& img) {
    // cleanup
    clear_internals();
    
    // makes sure the image is normalized
    cv::normalize(img, norm, 0.0, 255.0, cv::NORM_MINMAX, CV_32F);

    // extracts the magnitude and orientations images
    magnitude_and_orientation(img, mag, ori);

    // iterates over all blocks and cells
    for (size_t y = 0; y <= mag.rows - _blocksize; y += _stride) {
        for (size_t x = 0; x <= mag.cols - _blocksize; x += _stride) {
            cv::Rect block_rect = cv::Rect(x, y, _blocksize, _blocksize);
            THist block_hist = process_block(cv::Mat(mag, block_rect), cv::Mat(ori, block_rect));

            // concatenate all the blocks histograms
            img_hist.insert(std::end(img_hist), std::begin(block_hist), std::end(block_hist));
        }
    }

    return img_hist;
}

void HOG::magnitude_and_orientation(const cv::Mat& img, cv::Mat& mag, cv::Mat& ori) {
    cv::Mat Dx, Dy;
    cv::filter2D(img, Dx, CV_32F, _kernelx);
    cv::filter2D(img, Dy, CV_32F, _kernely);
    cv::magnitude(Dx, Dy, mag);
    cv::phase(Dx, Dy, ori, true);
}

HOG::THist HOG::process_block(const cv::Mat& block_mag, const cv::Mat& block_ori) {
    HOG::THist block_hist_concat;

    for (size_t y = 0; y < block_mag.rows; y += _cellsize) {
        for (size_t x = 0; x < block_mag.cols; x += _cellsize) {
            //std::cout << "Cell x:" << x << ", y:" << y << "\n";
            cv::Rect cell_rect = cv::Rect(x, y, _cellsize, _cellsize);
            cv::Mat cell_mag = cv::Mat(block_mag, cell_rect);
            cv::Mat cell_ori = cv::Mat(block_ori, cell_rect);
            HOG::THist cell_hist = process_cell(cell_mag, cell_ori);
            block_hist_concat.insert(std::end(block_hist_concat), std::begin(cell_hist), std::end(cell_hist));
        }
    }

    _block_norm(block_hist_concat); // inplace normalization
    _all_hists.push_back(block_hist_concat);
    return block_hist_concat;
}

HOG::THist HOG::process_cell(const cv::Mat& cell_mag, const cv::Mat& cell_ori) {
    HOG::THist cell_hist(_binning);
    if(_grad_type == GRADIENT_SIGNED) {
        for (size_t i = 0; i < cell_mag.rows; ++i) {
            const HOG::TType* ptr_row_mag = cell_mag.ptr<HOG::TType>(i);
            const HOG::TType* ptr_row_ori = cell_ori.ptr<HOG::TType>(i);
            for (size_t j = 0; j < cell_mag.cols; ++j) {
                cell_hist[static_cast<int>(ptr_row_ori[j] / _bin_width)] += ptr_row_mag[j];
            }
        }
    } else {
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

    // retrieve the max value of the final HOG histogram
    HOG::TType hist_max = *std::max_element(std::begin(img_hist), std::end(img_hist));
    //HOG::TType hist_mean = std::accumulate(std::begin(img_hist), std::end(img_hist), 0.0)/img_hist.size();

    // iterates over all blcoks and cells
    size_t idx_blocks = 0;
    size_t y, x, i, j;

    for (y = 0; y <= norm.rows - _blocksize; y += _stride) {
        for (x = 0; x <= norm.cols - _blocksize; x += _stride) {
            size_t idx_cells = 0;

            for (i = 0; i < _blocksize; i += _cellsize) {
                for (j = 0; j < _blocksize; j += _cellsize) {
                    // retrieves the cell histogram
                    HOG::THist block_hist = _all_hists[idx_blocks];
                    HOG::THist cell_hist(_binning);
                    std::copy(  std::begin(block_hist) + idx_cells * _binning,
                                std::begin(block_hist) + (idx_cells + 1)*_binning,
                                std::begin(cell_hist) );

                    // retrieve the max value of the this cell histogram
                    HOG::TType max = *std::max_element(std::begin(cell_hist), std::end(cell_hist));
                    //std::cout << "max=" << max << "\n";
                    //HOG::TType mean = std::accumulate(std::begin(cell_hist), std::end(cell_hist), 0.0)/cell_hist.size();
                    //std::cout << "mean=" << mean << "\n";

                    int color_magnitude = static_cast<int>(max / hist_max * 255.0);
                    //std::cout << "color_magnitude=" << color_magnitude << "\n";

                    // iterates over the cell histogram
                    for (size_t k = 0; k < cell_hist.size(); ++k) {
                        // fixed line thinkness
                        int thickness = 2;

                        // length of the "arrows"
                        int length = static_cast<int>((cell_hist[k] / max) * _cellsize / 2);

                        if (length > 0 && !isinf(length)) {
                            // draw "arrows" of varing length
                            if(_grad_type == GRADIENT_SIGNED) {
                                cv::line(vector_mask, cv::Point(x + j + _cellsize / 2, y + i + _cellsize / 2),
                                     cv::Point(  x + j + _cellsize / 2 + cos((k * _bin_width) * 3.1415 / 180)*length,
                                                 y + i + _cellsize / 2 + sin((k * _bin_width) * 3.1415 / 180)*length),
                                     cv::Scalar(color_magnitude, color_magnitude, color_magnitude), thickness);
                            } else {
                                cv::line(vector_mask, 
                                    cv::Point(  x + j + _cellsize / 2 + cos((k * _bin_width+180) * 3.1415 / 180)*length,
                                                 y + i + _cellsize / 2 + sin((k * _bin_width+180) * 3.1415 / 180)*length),
                                     cv::Point(  x + j + _cellsize / 2 + cos((k * _bin_width) * 3.1415 / 180)*length,
                                                 y + i + _cellsize / 2 + sin((k * _bin_width) * 3.1415 / 180)*length),
                                     cv::Scalar(color_magnitude, color_magnitude, color_magnitude), thickness);
                            }
                        }
                    }

                    // draw cell delimiters
                    cv::line(vector_mask, cv::Point(x + i - 1, y + j - 1), cv::Point(x + i + norm.rows - 1, y + j - 1), cv::Scalar(255, 255, 255), 1);
                    cv::line(vector_mask, cv::Point(x + i - 1, y + j - 1), cv::Point(x + i - 1, y + j + norm.rows - 1), cv::Scalar(255, 255, 255), 1);
                    idx_cells++;
                }

                // draw cell delimiters
                cv::line(vector_mask, cv::Point(x + i - 1, y + j - 1), cv::Point(x + i + norm.rows - 1, y + j - 1), cv::Scalar(255, 255, 255), 1);
                cv::line(vector_mask, cv::Point(x + i - 1, y + j - 1), cv::Point(x + i - 1, y + j + norm.rows - 1), cv::Scalar(255, 255, 255), 1);
            }

            // draw cell delimiters
            cv::line(vector_mask, cv::Point(x + i - 1, y + j - 1), cv::Point(x + i + norm.rows - 1, y + j - 1), cv::Scalar(255, 255, 255), 1);
            cv::line(vector_mask, cv::Point(x + i - 1, y + j - 1), cv::Point(x + i - 1, y + j + norm.rows - 1), cv::Scalar(255, 255, 255), 1);
            idx_blocks++;
        }

        // draw cell delimiters
        cv::line(vector_mask, cv::Point(x - 1, y - 1), cv::Point(x + norm.rows - 1, y - 1), cv::Scalar(255, 255, 255), 1);
        cv::line(vector_mask, cv::Point(x - 1, y - 1), cv::Point(x - 1, y + norm.rows - 1), cv::Scalar(255, 255, 255), 1);
    }

    return vector_mask;
}

void HOG::clear_internals() {
    img_hist.clear();
    for(auto& h:_all_hists) 
        h.clear();
    _all_hists.clear();
}
