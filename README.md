# cuda-median-filter
 C++ median filter for images using OpenCV, with CUDA for GPU acceleration


# openCV installation

## install CV ubuntu
```console
sudo apt update
sudo apt install libopencv-dev
```
## run test

```
mkdir -p ./out && g++ -O3 ./source/test_opencv.cpp -o ./out/test_opencv `pkg-config --cflags --libs opencv4` && ./out/test_opencv && rm out/test_opencv
```

```console
nvcc -o median_filter median_filter.cu `pkg-config --cflags --libs opencv4`
```




nvcc -O3 -o main main_cuda.cu `pkg-config --cflags --libs opencv4` && ./main image2.png out9.png 9