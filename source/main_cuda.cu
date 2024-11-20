#include "median_filter_cuda.h"

#include <iostream>
#include <chrono>

int main(int argc, char const *argv[])
{
    auto t_start = std::chrono::high_resolution_clock::now();

    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image> <filter_size>" << std::endl;
        return -1;
    }

    std::string input_file_name = argv[1];
    std::string output_file_name = argv[2];
    int filter_size = atoi(argv[3]);

    cv::Mat input_image = cv::imread(input_file_name, cv::IMREAD_GRAYSCALE);
    if (input_image.empty()) {
        std::cerr << "Error: could not read input image." << std::endl;
        return -1;
    }

    //+: print input image pixel if you want

    cv::Mat output_image(input_image.size(), input_image.type());
    gray_median_filter_cuda(input_image, output_image, filter_size);

    if (!cv::imwrite(output_file_name, output_image)) {
        std::cerr << "Error: could not save output image." << std::endl;
        return -1;
    }
    std::cout << "output image saved to " << output_file_name << std::endl;

    auto t_end = std::chrono::high_resolution_clock::now();
    auto t_duration = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start);
    std::cout << "execution time: " << t_duration.count() << " us" << std::endl;
    //+: print output image pixel if you want


    return 0;
}
