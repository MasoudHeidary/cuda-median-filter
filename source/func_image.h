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

bool image_identical(const cv::Mat& image1, const cv::Mat& image2) {
    if (image1.size() != image2.size() || image1.type() != image2.type()) {
        std::cout << "Images differ in size or type." << std::endl;
        return false;
    }

    cv::Mat diff;
    cv::absdiff(image1, image2, diff);

    // Check if all pixels are identical
    if (cv::countNonZero(diff) == 0) {
        return true;
    }

    // Log the coordinates of differing pixels
    for (int row = 0; row < diff.rows; ++row) {
        for (int col = 0; col < diff.cols; ++col) {
            // Check for non-zero pixel values
            if (diff.at<uchar>(row, col) != 0) {
                // std::cout << "Difference at (" << row << ", " << col << ")" << std::endl;
            }
        }
    }

    return false;
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
