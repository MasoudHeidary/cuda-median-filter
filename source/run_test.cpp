#include "__test_opencv.h"
#include "func_image.h"
#include "median_filter_cpu.h"

#include <iostream>
#include <cstdio>

#define DELETE_TMP_FILE true
#define FILTER_SIZE 3
#define SMALL_IMAGE_NAME "test_image.png"
#define SMALL_IMAGE_WIDTH 4
#define SAMLL_IMAGE_HEIGH 4 

int main(int argc, char const *argv[])
{
    std::cout << "openCV version: " << get_opencv_version() << std::endl;
    
    std::string image_name = "test_image.png";
    std::cout << "generating a small random gray image:" << std::endl;
    cv::Mat smlimage = generate_gray_image(SMALL_IMAGE_NAME, SMALL_IMAGE_WIDTH, SAMLL_IMAGE_HEIGH);
    print_gray_pixel_value(smlimage);

    std::cout << "filter size: " << FILTER_SIZE << std::endl;
    std::cout << "applying gray median filter on the image:" << std::endl;
    smlimage = gray_median_filter(smlimage, FILTER_SIZE);
    print_gray_pixel_value(smlimage);


    // run on cpu
    // run on cuda
    // comprare time and check to be same
    std::remove(SMALL_IMAGE_NAME);
    

    return 0;
}
