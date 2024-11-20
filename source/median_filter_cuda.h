#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>

#include "func_image.h"

#define FILTER_SIZE 3
#define BLOCK_SIZE 16 

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

        while (j >= 0 && window[j] > key) {
            window[j + 1] = window[j];
            j--;
        }
        window[j + 1] = key;
    }
}


__device__ void parallelSort(unsigned char* window, int size, int threadIdx) {
    for (int k = 2; k <= size; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            int idx = threadIdx;

            if (idx < size) {
                int ixj = idx ^ j;

                // Only compare and swap if within bounds and direction is valid
                if (ixj > idx) {
                    if ((idx & k) == 0) { // Ascending sort
                        if (window[idx] > window[ixj]) {
                            // Swap
                            unsigned char temp = window[idx];
                            window[idx] = window[ixj];
                            window[ixj] = temp;
                        }
                    } else { // Descending sort
                        if (window[idx] < window[ixj]) {
                            // Swap
                            unsigned char temp = window[idx];
                            window[idx] = window[ixj];
                            window[ixj] = temp;
                        }
                    }
                }
            }
            __syncthreads(); // Synchronize threads in the block
        }
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
        int leftX = max(x - padSize, 0); // Clamp to the left edge
        sharedMem[sharedY * sharedWidth + threadIdx.x] = inputImage[clampedY * width + leftX];

        // Right padding
        int rightX = min(x + blockDim.x, width - 1); // Clamp to the right edge
        sharedMem[sharedY * sharedWidth + sharedX + blockDim.x] = inputImage[clampedY * width + rightX];
    }

    if (threadIdx.y < padSize) {
        // Top padding
        int topY = max(y - padSize, 0); // Clamp to the top edge
        sharedMem[threadIdx.y * sharedWidth + sharedX] = inputImage[topY * width + clampedX];

        // Bottom padding
        int bottomY = min(y + blockDim.y, height - 1); // Clamp to the bottom edge
        sharedMem[(sharedY + blockDim.y) * sharedWidth + sharedX] = inputImage[bottomY * width + clampedX];
    }

    // Handle corner cases (optional, but ensures correctness for large padding)
    if (threadIdx.x < padSize && threadIdx.y < padSize) {
        // Top-left corner
        sharedMem[threadIdx.y * sharedWidth + threadIdx.x] = inputImage[max(y - padSize, 0) * width + max(x - padSize, 0)];
        // Top-right corner
        sharedMem[threadIdx.y * sharedWidth + sharedX + blockDim.x] = inputImage[max(y - padSize, 0) * width + min(x + blockDim.x, width - 1)];
        // Bottom-left corner
        sharedMem[(sharedY + blockDim.y) * sharedWidth + threadIdx.x] = inputImage[min(y + blockDim.y, height - 1) * width + max(x - padSize, 0)];
        // Bottom-right corner
        sharedMem[(sharedY + blockDim.y) * sharedWidth + sharedX + blockDim.x] = inputImage[min(y + blockDim.y, height - 1) * width + min(x + blockDim.x, width - 1)];
    }

    __syncthreads();

    // Apply the median filter
    if (x < width && y < height) {
        // unsigned char window[FILTER_SIZE * FILTER_SIZE];
        unsigned char window[1024];

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




void gray_median_filter_cuda(const cv::Mat& inputImage, cv::Mat& outputImage, int filterSize) {
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

int color_median_filter_cuda(const cv::Mat& inputImage, cv::Mat& outputImage, const int filterSize) {
    if (inputImage.channels() != 3) {
        // error code
        return -1;
    }

    
    const int width = inputImage.cols;
    const int height = inputImage.rows;
    const int imageSize = width * height;
    const int channelSize = imageSize; // Size of a single channel

    // Allocate memory for each channel
    unsigned char *d_inputR, *d_inputG, *d_inputB;
    unsigned char *d_outputR, *d_outputG, *d_outputB;

    cudaMalloc(&d_inputR, channelSize);
    cudaMalloc(&d_inputG, channelSize);
    cudaMalloc(&d_inputB, channelSize);
    cudaMalloc(&d_outputR, channelSize);
    cudaMalloc(&d_outputG, channelSize);
    cudaMalloc(&d_outputB, channelSize);

    // Split the image into its respective channels
    std::vector<cv::Mat> channels(3);
    cv::split(inputImage, channels);

    // Copy each channel to device memory
    cudaMemcpy(d_inputR, channels[0].data, channelSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputG, channels[1].data, channelSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputB, channels[2].data, channelSize, cudaMemcpyHostToDevice);

    // Kernel launch parameters
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    size_t sharedMemSize = (BLOCK_SIZE + 2 * (filterSize / 2)) * (BLOCK_SIZE + 2 * (filterSize / 2)) * sizeof(unsigned char);

    // Apply median filter to each channel
    medianFilterKernel<<<gridDim, blockDim, sharedMemSize>>>(d_inputR, d_outputR, width, height, filterSize);
    medianFilterKernel<<<gridDim, blockDim, sharedMemSize>>>(d_inputG, d_outputG, width, height, filterSize);
    medianFilterKernel<<<gridDim, blockDim, sharedMemSize>>>(d_inputB, d_outputB, width, height, filterSize);

    // Copy the filtered channels back to host memory
    cudaMemcpy(channels[0].data, d_outputR, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(channels[1].data, d_outputG, channelSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(channels[2].data, d_outputB, channelSize, cudaMemcpyDeviceToHost);

    // Merge the filtered channels back into a single image
    cv::merge(channels, outputImage);

    // Free device memory
    cudaFree(d_inputR);
    cudaFree(d_inputG);
    cudaFree(d_inputB);
    cudaFree(d_outputR);
    cudaFree(d_outputG);
    cudaFree(d_outputB);

    return 0;
}




// void color_median_filter_cuda(const cv::Mat& inputImage, cv::Mat& outputImage, int filterSize) {
//     // Ensure input is a color image
//     if (inputImage.channels() != 3) {
//         std::cerr << "Input image is not a color image!" << std::endl;
//         return;
//     }

//     const int width = inputImage.cols;
//     const int height = inputImage.rows;
//     const int imageSize = width * height;
//     const int channelSize = imageSize; // Size of a single channel

//     // Allocate memory for each channel
//     unsigned char *d_inputR, *d_inputG, *d_inputB;
//     unsigned char *d_outputR, *d_outputG, *d_outputB;

//     cudaMalloc(&d_inputR, channelSize);
//     cudaMalloc(&d_inputG, channelSize);
//     cudaMalloc(&d_inputB, channelSize);
//     cudaMalloc(&d_outputR, channelSize);
//     cudaMalloc(&d_outputG, channelSize);
//     cudaMalloc(&d_outputB, channelSize);

//     // Split the image into its respective channels
//     std::vector<cv::Mat> channels(3);
//     cv::split(inputImage, channels);

//     // Copy each channel to device memory
//     cudaMemcpy(d_inputR, channels[0].data, channelSize, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_inputG, channels[1].data, channelSize, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_inputB, channels[2].data, channelSize, cudaMemcpyHostToDevice);

//     // Kernel launch parameters
//     dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
//     dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

//     size_t sharedMemSize = (BLOCK_SIZE + 2 * (filterSize / 2)) * (BLOCK_SIZE + 2 * (filterSize / 2)) * sizeof(unsigned char);

//     // Apply median filter to each channel
//     medianFilterKernel<<<gridDim, blockDim, sharedMemSize>>>(d_inputR, d_outputR, width, height, filterSize);
//     medianFilterKernel<<<gridDim, blockDim, sharedMemSize>>>(d_inputG, d_outputG, width, height, filterSize);
//     medianFilterKernel<<<gridDim, blockDim, sharedMemSize>>>(d_inputB, d_outputB, width, height, filterSize);

//     // Copy the filtered channels back to host memory
//     cudaMemcpy(channels[0].data, d_outputR, channelSize, cudaMemcpyDeviceToHost);
//     cudaMemcpy(channels[1].data, d_outputG, channelSize, cudaMemcpyDeviceToHost);
//     cudaMemcpy(channels[2].data, d_outputB, channelSize, cudaMemcpyDeviceToHost);

//     // Merge the filtered channels back into a single image
//     cv::merge(channels, outputImage);

//     // Free device memory
//     cudaFree(d_inputR);
//     cudaFree(d_inputG);
//     cudaFree(d_inputB);
//     cudaFree(d_outputR);
//     cudaFree(d_outputG);
//     cudaFree(d_outputB);
// }
