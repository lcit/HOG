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
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include <functional>
#include <math.h>

class HOG {   
public:
    using TType = float;
    using THist = std::vector<TType>;
    
    static const size_t GRADIENT_SIGNED = 360; 
    static const size_t GRADIENT_UNSIGNED = 180; 
    static constexpr TType epsilon = 1e-6;

    // see: https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients#Block_normalization
    static void L1norm(THist& v);
    static void L1sqrt(THist& v);
    static void L2norm(THist& v);
    static void L2hys(THist& v);
    static void none(THist& v);

private:
    const size_t _blocksize;
    const size_t _cellsize;
    const size_t _stride;
    const size_t _grad_type; ///< "signed" (0..360) or "unsigned" (0..180) gradient
    const size_t _binning; ///< the number of bins for each cell-histogram
    const size_t _bin_width; ///< size of one bin in degree
    const size_t _n_cells_per_block_y = _blocksize/_cellsize;
    const size_t _n_cells_per_block_x = _n_cells_per_block_y;
    const size_t _n_cells_per_block = _n_cells_per_block_y*_n_cells_per_block_x;
    const size_t _stride_unit = _stride/_cellsize;
    const unsigned _n_threads;
    const std::function<void(THist&)> _block_norm;  ///< function that normalize the block histogram
    const cv::Mat _kernelx = (cv::Mat_<char>(1, 3) << -1, 0, 1); ///< derivive kernel
    const cv::Mat _kernely = (cv::Mat_<char>(3, 1) << -1, 0, 1); ///< derivive kernel
    size_t _n_cells_y;
    size_t _n_cells_x;

    cv::Mat mag, ori, norm;
    THist img_hist;
    std::vector<THist> _all_hists;

    std::vector<std::vector<THist>> _cell_hists;
    

public:
    HOG(const size_t blocksize,
        std::function<void(THist&)> block_norm = L2hys, const unsigned n_threads = 1);
    HOG(const size_t blocksize, size_t cellsize,
        std::function<void(THist&)> block_norm = L2hys, const unsigned n_threads = 1);
    HOG(const size_t blocksize, size_t cellsize, size_t stride,
        std::function<void(THist&)> block_norm = L2hys, const unsigned n_threads = 1);
    HOG(const size_t blocksize, size_t cellsize, size_t stride, size_t binning = 9, 
        size_t grad_type = GRADIENT_UNSIGNED, std::function<void(THist&)> block_norm = L2hys, const unsigned n_threads = 1);
    ~HOG();

    /// Retrieves the HOG from an image
    ///
    /// @param img: source image (any size)
    /// @return the HOG histogram as std::vector
    void process(const cv::Mat& img);
    
    THist retrieve(const cv::Rect& rect);

private:
    /// Retrieves magnitude and orientation form an image
    ///
    /// @param img: source image (any size)
    /// @param mag: ref. to the magnitude matrix where to store the result
    /// @param pri: ref. to the orientation matrix where to store the result
    /// @return none
    void magnitude_and_orientation(const cv::Mat& img);

    /// Iterates over a block and concatenates the cell histograms
    ///
    /// @param block_mag: a portion (block) of the magnitude matrix
    /// @param block_ori: a portion (block) of the orientation matrix
    /// @return the block histogram as std::vector
    THist process_block(const cv::Mat& block_mag, const cv::Mat& block_ori);

    /// Iterates over a cell to create the cell histogram
    ///
    /// @param cell_mag: a portion of a block (cell) of the magnitude matrix
    /// @param cell_ori: a portion of a block (cell) of the orientation matrix
    /// @return the cell histogram as std::vector
    THist process_cell(const cv::Mat& cell_mag, const cv::Mat& cell_ori);
    
    /// Clear internal/local data
    ///
    /// @param none
    /// @return none
    void clear_internals();

public:
    /// Utility funtion to retreve the magnitude matrix
    ///
    /// @return the magnitude matrix CV_32F
    cv::Mat get_magnitudes();

    /// Utility funtion to retreve the orientation matrix
    ///
    /// @return the orientation matrix CV_32F
    cv::Mat get_orientations();

    /// Utility funtion to retreve a mask of vectors
    ///
    /// @return the vector matrix CV_32F
    cv::Mat get_vector_mask();
};
