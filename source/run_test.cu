#include "__test_opencv.h"
#include "func_image.h"
#include "median_filter_cpu.h"
#include "median_filter_cuda.h"

#include <iostream>
#include <cstdio>
#include <chrono>
#include <cmath>

#define DELETE_TMP_FILE true
#define TEST_FILTER_SIZE 5
#define IMAGE_NAME "test_image.png"
#define IMAGE_WIDTH 4
#define IMAGE_HEIGH 4 
// #define LARGE_IMAGE_SCALE 5
#define LARGE_IMAGE_SCALE pow(2,11)
#define PRINT_LARGE_IMAGE false


cv::Mat gray_median_filter_cuda(const cv::Mat& inputImage, int filterSize);

int main(int argc, char const *argv[])
{
    std::cout << "openCV version: " << get_opencv_version() << std::endl;
    
    std::cout << "generating a small random gray image:" << std::endl;
    cv::Mat smlimage = generate_gray_image(IMAGE_NAME, IMAGE_WIDTH, IMAGE_HEIGH);
    print_gray_pixel_value(smlimage);

    std::cout << "filter size: " << TEST_FILTER_SIZE << std::endl;
    std::cout << "applying gray median filter on the image:" << std::endl;
    smlimage = gray_median_filter(smlimage, TEST_FILTER_SIZE);
    print_gray_pixel_value(smlimage);

    std::cout << "generating a large random gray image" << std::endl;
    cv::Mat lrgimage = generate_gray_image(IMAGE_NAME, IMAGE_WIDTH*LARGE_IMAGE_SCALE, IMAGE_HEIGH*LARGE_IMAGE_SCALE);
    if (PRINT_LARGE_IMAGE) {
        std::cout << "LARGE Image:" << std::endl;
        print_gray_pixel_value(lrgimage);
        std::cout << "====================================" << std::endl;
    }

    // timing
    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // ==================== CPU ====================
    start = std::chrono::high_resolution_clock::now();
    cv::Mat cpu_image = gray_median_filter(lrgimage, TEST_FILTER_SIZE);
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto cpu_time = duration.count();
    std::cout << "median filter on cpu: " << duration.count() << "\tms" << std::endl;
    
    if (PRINT_LARGE_IMAGE) {
        std::cout << "CPU output:" << std::endl;
        print_gray_pixel_value(cpu_image);
        std::cout << "====================================" << std::endl;
    }

    // ==================== CUDA ====================
    start = std::chrono::high_resolution_clock::now();
    cv::Mat cuda_image(lrgimage.size(), lrgimage.type());
    gray_median_filter_cuda(lrgimage, cuda_image, TEST_FILTER_SIZE);
    end = std::chrono::high_resolution_clock::now();
    
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    auto cuda_time = duration.count();
    std::cout << "median filter on cuda: " << duration.count() << "\tms" << std::endl;
    
    if(PRINT_LARGE_IMAGE) {
        std::cout << "CUDA output:" << std::endl;
        print_gray_pixel_value(cuda_image);
        std::cout << "====================================" << std::endl;
    }


    std::cout << "matching pictures test? [" << (image_identical(cpu_image, cuda_image)? "TRUE":"FALSE") << "]" << std::endl;
    std::cout << "speed up: " << cpu_time/cuda_time << "x" << std::endl;


    std::remove(IMAGE_NAME);
    

    return 0;
}
