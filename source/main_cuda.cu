#include "median_filter_cuda.h"

#include <iostream>
#include <chrono>

int main(int argc, char const *argv[])
{
    auto t_start = std::chrono::high_resolution_clock::now();

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image> <filter_size> <colorful:true|false>" << std::endl;
        return -1;
    }

    std::string input_file_name = argv[1];
    std::string output_file_name = argv[2];
    int filter_size = 3;
    if (argc >= 4) 
        filter_size = atoi(argv[3]);
    int is_colorful = false;
    if (argc >= 5) {
        std::string color_arg = argv[4];
        is_colorful = (color_arg == "true" ? true : false);
    }


    cv::Mat input_image = cv::imread(input_file_name, is_colorful ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE);
    if (input_image.empty()) {
        std::cerr << "Error: could not read input image." << std::endl;
        return -1;
    }

    //+: print input image pixel if you want

    cv::Mat output_image(input_image.size(), input_image.type());
    if (is_colorful == false)
        gray_median_filter_cuda(input_image, output_image, filter_size);
    else
        color_median_filter_cuda(input_image, output_image, filter_size);

    if (!cv::imwrite(output_file_name, output_image)) {
        std::cerr << "Error: could not save output image." << std::endl;
        return -1;
    }
    std::cout << "output image saved to " << output_file_name << std::endl;

    auto t_end = std::chrono::high_resolution_clock::now();
    auto t_duration = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
    std::cout << "execution time: " << t_duration.count() << " us" << std::endl;
    
    //+: print output image pixel if you want
    // print_gray_pixel_value(output_image);

    return 0;
}
