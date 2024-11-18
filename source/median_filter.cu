#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>

#include "_test_image.h"

#define FILTER_SIZE 3
#define BLOCK_SIZE 16  // Adjust as needed

// bubble sort
// __device__ void sortWindow(unsigned char* window, int size) {
//     for (int i = 0; i < size - 1; i++) {
//         for (int j = 0; j < size - i - 1; j++) {
//             if (window[j] > window[j + 1]) {
//                 unsigned char temp = window[j];
//                 window[j] = window[j + 1];
//                 window[j + 1] = temp;
//             }
//         }
//     }
// }

//insertion sort
__device__ void sortWindow(unsigned char* window, int size) {
    for (int i = 1; i < size; i++) {
        unsigned char key = window[i];
        int j = i - 1;

        // Move elements of window[0..i-1], that are greater than key,
        // one position ahead of their current position
        while (j >= 0 && window[j] > key) {
            window[j + 1] = window[j];
            j--;
        }
        window[j + 1] = key;
    }
}

// global memory access median filter
// __global__ void medianFilterKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int filterSize) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     int padSize = filterSize / 2;

//     // Ensure threads within the image bounds
//     if (x < width && y < height) {
//         unsigned char window[FILTER_SIZE * FILTER_SIZE];

//         // Collect the filter window
//         for (int ky = -padSize; ky <= padSize; ky++) {
//             for (int kx = -padSize; kx <= padSize; kx++) {
//                 int nx = min(max(x + kx, 0), width - 1); // Clamp to image boundaries
//                 int ny = min(max(y + ky, 0), height - 1); // Clamp to image boundaries
//                 window[(ky + padSize) * filterSize + (kx + padSize)] = inputImage[ny * width + nx];
//             }
//         }

//         // Sort the window to find the median
//         sortWindow(window, filterSize * filterSize);

//         // Assign the median value to the output image
//         outputImage[y * width + x] = window[(filterSize * filterSize) / 2];
//     }
// }


// shared memory optimized
__global__ void medianFilterKernel(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int filterSize) {
    extern __shared__ unsigned char sharedMem[];

    int padSize = filterSize / 2;

    // Shared memory dimensions
    int sharedWidth = blockDim.x + 2 * padSize;

    // Global thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Shared memory coordinates
    int sharedX = threadIdx.x + padSize;
    int sharedY = threadIdx.y + padSize;

    // Clamping global indices to valid image range
    int clampedX = min(max(x, 0), width - 1);
    int clampedY = min(max(y, 0), height - 1);

    // Load pixel data into shared memory
    sharedMem[sharedY * sharedWidth + sharedX] = inputImage[clampedY * width + clampedX];

    // Load padding
    if (threadIdx.x < padSize) {
        // Left padding
        int leftX = max(clampedX - padSize, 0);
        sharedMem[sharedY * sharedWidth + threadIdx.x] = inputImage[clampedY * width + leftX];

        // Right padding
        int rightX = min(clampedX + blockDim.x, width - 1);
        sharedMem[sharedY * sharedWidth + sharedX + blockDim.x] = inputImage[clampedY * width + rightX];
    }

    if (threadIdx.y < padSize) {
        // Top padding
        int topY = max(clampedY - padSize, 0);
        sharedMem[threadIdx.y * sharedWidth + sharedX] = inputImage[topY * width + clampedX];

        // Bottom padding
        int bottomY = min(clampedY + blockDim.y, height - 1);
        sharedMem[(sharedY + blockDim.y) * sharedWidth + sharedX] = inputImage[bottomY * width + clampedX];
    }

    __syncthreads();

    // Apply the median filter
    if (x < width && y < height) {
        unsigned char window[FILTER_SIZE * FILTER_SIZE];

        // Collect the filter window from shared memory
        for (int ky = -padSize; ky <= padSize; ky++) {
            for (int kx = -padSize; kx <= padSize; kx++) {
                window[(ky + padSize) * filterSize + (kx + padSize)] =
                    sharedMem[(sharedY + ky) * sharedWidth + (sharedX + kx)];
            }
        }

        // Sort the window to find the median
        sortWindow(window, filterSize * filterSize);

        // Write the median value to the output image
        outputImage[y * width + x] = window[(filterSize * filterSize) / 2];
    }
}


void applyMedianFilterCUDA(const cv::Mat& inputImage, cv::Mat& outputImage, int filterSize) {
    const int width = inputImage.cols;
    const int height = inputImage.rows;
    const int imageSize = width * height;

    unsigned char *d_inputImage, *d_outputImage;

    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);

    cudaMemcpy(d_inputImage, inputImage.data, imageSize, cudaMemcpyHostToDevice);

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    size_t sharedMemSize = (BLOCK_SIZE + 2 * (filterSize / 2)) * (BLOCK_SIZE + 2 * (filterSize / 2)) * sizeof(unsigned char);

    medianFilterKernel<<<gridDim, blockDim, sharedMemSize>>>(d_inputImage, d_outputImage, width, height, filterSize);

    cudaMemcpy(outputImage.data, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image>" << std::endl;
        return -1;
    }

    std::string inputFilename = argv[1];
    std::string outputFilename = argv[2];

    cv::Mat inputImage = cv::imread(inputFilename, cv::IMREAD_GRAYSCALE);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not read input image." << std::endl;
        return -1;
    }
    std::cout << "input image:" << std::endl;
    printPixelValues(inputImage);

    cv::Mat outputImage(inputImage.size(), inputImage.type());

    applyMedianFilterCUDA(inputImage, outputImage, FILTER_SIZE);

    if (!cv::imwrite(outputFilename, outputImage)) {
        std::cerr << "Error: Could not save output image." << std::endl;
        return -1;
    }

    std::cout << "output image:" << std::endl;
    printPixelValues(outputImage);

    std::cout << "Output image saved to " << outputFilename << std::endl;

    return 0;
}
