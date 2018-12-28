# HOG

C++ - Simple CPU implementation of the HOG (Histogram of Oriented Grandients) based on OpenCV's utility functions.

The reference article: https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf

### Prerequisites

OpenCV 3

### Run example
```
./clean.sh; ./build.sh
./run_person.sh
```

### Example usage

```C++
int main(int argc, char* argv[]){

  // Open an image
  cv::Mat image = cv::imread(argv[1], CV_8U);

  // Set up the HOG object
  size_t cellsize = 8;
  size_t blocksize = cellsize*2;
  size_t stride = cellsize;
  size_t binning = 9;
  HOG hog(blocksize, cellsize, stride, binning, HOG::GRADIENT_UNSIGNED, HOG::L2hys);

  // example how to save and load a HOG model
  hog.save("hog.ext");
  hog = HOG::load("hog.ext");

  // Process the whole image
  hog.process(image);

  // Retrieve HOG from a ROI/window/sub-image
  cv::Size window(50,100);

  #pragma omp parallel num_threads(8)
  {
    #pragma omp for collapse(2)
    for(int x=0; x<image.cols-window.width; x += cellsize){
      for(int y=0; y<image.rows-window.height; y += cellsize){
        cv::Rect roi = cv::Rect(x,y, window.width, window.height);
        auto hist = hog.retrieve(roi);

        // Print resulting histograms
        std::cout << "Histogram size: " << hist.size() << "\n";
        for(auto h:hist)
          std::cout << h << ",";
        std::cout << "\n";
      }
    }
  }
}
```

![alt tag](https://raw.githubusercontent.com/lcit/HOG/master/img/HOG.png)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
