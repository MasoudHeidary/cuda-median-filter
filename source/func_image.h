#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>

void print_gray_pixel_value(const cv::Mat& image) {
    for (int row = 0; row < image.rows; ++row) {
        for (int col = 0; col < image.cols; ++col) {
            std::cout << static_cast<int>(image.at<uchar>(row, col)) << "\t";
        }
        std::cout << std::endl; // Newline for each row
    }
}

cv::Mat generate_gray_image(std::string filename, int width, int height) {
    cv::Mat generatedImage(height, width, CV_8U); // Grayscale image (single channel)

    cv::RNG rng(cv::getTickCount());
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            generatedImage.at<uchar>(row, col) = rng.uniform(0, 256);;
        }
    }

    // error checking ignored
    cv::imwrite(filename, generatedImage);
    
    return generatedImage;
}
