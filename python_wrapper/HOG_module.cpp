/*  ==========================================================================================
    Author: Leonardo Citraro
    Company:
    Filename: HOG_module.hpp
    Last modifed:   28.12.2016 by Leonardo Citraro
    Description:    Python wrapper using C API

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
#include <Python.h>
#include </usr/local/lib/python2.7/dist-packages/numpy/core/include/numpy/arrayobject.h>
#include "HOG.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

static PyObject* HOG_func(PyObject *dummy, PyObject *args){

    int blocksize;
    int cellsize;
    int stride;
    int binning;
    int grad_type_i, grad_type;
    int block_norm_i;
    std::function<void(HOG::THist&)> block_norm;
    PyObject *arg1 = NULL;

    if (!PyArg_ParseTuple(args, "iiiiiiO", &blocksize, &cellsize, &stride, &binning, &grad_type_i, &block_norm_i, &arg1)) 
        return NULL;

    int** image;
    PyArray_Descr *descr;
    descr = PyArray_DescrFromType(NPY_INT);
    npy_intp dims_image[3];
    if (PyArray_AsCArray(&arg1, (void **)&image, dims_image, 2, descr) < 0) {
        PyErr_SetString(PyExc_TypeError, "error converting to c array");
        return NULL;
    }

    switch(grad_type_i){
        case 0: 
            grad_type = HOG::GRADIENT_SIGNED; break;
        case 1: 
            grad_type = HOG::GRADIENT_UNSIGNED; break;
        default: 
            return NULL;
    }
    switch(block_norm_i){
        case 0: 
            block_norm = HOG::none; break;
        case 1: 
            block_norm = HOG::L1norm; break;
        case 2: 
            block_norm = HOG::L1sqrt; break;
        case 3: 
            block_norm = HOG::L2norm; break;
        case 4: 
            block_norm = HOG::L2hys; break;
        default: 
            return NULL;
    }

    HOG hog(blocksize, cellsize, stride, binning, grad_type, block_norm);
    cv::Mat img = cv::Mat(dims_image[0], dims_image[1], CV_8U, image);
    hog.process(img);

    std::vector<float> hist = hog.retrieve(cv::Rect(0, 0, dims_image[1], dims_image[0]));

    float* temp = new float[hist.size()];
    std::memcpy(temp, hist.data(), hist.size());

    npy_intp dims_res[1] = {hist.size()};
    PyObject* arg2 = PyArray_SimpleNewFromData(1, dims_res, NPY_FLOAT, (void*)temp);
    PyArray_ENABLEFLAGS((PyArrayObject*)arg2, NPY_ARRAY_OWNDATA);

    return arg2;
}

/*  define functions in module */
static PyMethodDef HOG_method[] = 
{
     {"HOG_func", HOG_func, METH_VARARGS,
         "HOG"},
     {NULL, NULL, 0, NULL}
};

/* module initialization */
PyMODINIT_FUNC initHOG_module(void){
     (void) Py_InitModule("HOG_module", HOG_method);
     /* IMPORTANT: this must be called */
     import_array();
}
