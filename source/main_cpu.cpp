#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>

#define FILTER_SIZE 3

cv::Mat applyMedianFilter(const cv::Mat& inputImage, int filterSize = 3) {
    int padSize = filterSize / 2;
    cv::Mat paddedImage;
    
    cv::copyMakeBorder(inputImage, paddedImage, padSize, padSize, padSize, padSize, cv::BORDER_REPLICATE);

    cv::Mat outputImage = inputImage.clone();

    for (int y = padSize; y < paddedImage.rows - padSize; ++y) {
        for (int x = padSize; x < paddedImage.cols - padSize; ++x) {
            std::vector<uchar> window;

            for (int ky = -padSize; ky <= padSize; ++ky) {
                for (int kx = -padSize; kx <= padSize; ++kx) {
                    uchar pixelValue = paddedImage.at<uchar>(y + ky, x + kx);
                    window.push_back(pixelValue);
                }
            }

            std::sort(window.begin(), window.end());
            uchar medianValue = window[window.size() / 2];

            outputImage.at<uchar>(y - padSize, x - padSize) = medianValue;
        }
    }
    return outputImage;
}

int main(int argc, char const *argv[]) {

    if (argc < 3) {
        std::cerr << "arguments are not enough" << std::endl;
        return -1;
    }

    std::string input_filename = argv[1];
    std::string output_filename = argv[2];

    cv::Mat inputImage = cv::imread(input_filename, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error: image [" << input_filename << "] not find." << std::endl;
        return -1;
    }
    cv::Mat outputImage = applyMedianFilter(inputImage, FILTER_SIZE);

    if (cv::imwrite(output_filename, outputImage)) {
        std::cout << "Image saved successfully as " << output_filename << std::endl;
    } else {
        std::cerr << "Error: Could not save the image." << std::endl;
        return false;
    }
    
    // cv::imshow("Original Image", inputImage);
    // cv::imshow("Median Filtered Image", outputImage);
    // cv::waitKey(0);  // Wait for a key press before closing

    return 0;
}
