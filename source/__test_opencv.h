#include <opencv2/opencv.hpp>
#include <iostream>

std::string get_opencv_version(void) {
    return cv::getVersionString();
}