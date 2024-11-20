#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>

cv::Mat gray_median_filter(const cv::Mat& inputImage, int filterSize) {
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
