# cuda-median-filter
 C++ median filter for images using OpenCV, with CUDA for GPU acceleration


# openCV installation

## install CV ubuntu
```console
sudo apt update
sudo apt install libopencv-dev
```


## run test

compile and run the run_test.cu to check the speed up and functionality test

```console
nvcc -O3 -o out/run_test source/run_test.cu `pkg-config --cflags --libs opencv4` --disable-warnings && ./out/run_test
```

### output

```c++
openCV version: 4.6.0
generating a small random gray image:
87      59      43      180
74      151     112     137
110     187     82      85
78      163     58      160
filter size: 5
applying gray median filter on the image:
87      87      87      137
87      87      110     137
78      87      112     151
78      85      137     160
generating a large random gray image
median filter on cpu: 35102     ms
median filter on cuda: 189      ms
matching pictures test? [TRUE]
speed up: 185x
```


## use median filter

### compile

*## run all the commands on the main directory*      
compile the code       



```console
 nvcc -O3 -o out/main source/main_cuda.cu `pkg-config --cflags --libs opencv4` --disable-warnings
```

## using the filter

### original image

![image](doc/image.png)

### gray with filter size 3 -> gray3.png

```console
./out/main ./doc/image.png ./doc/gray3.png 3
```

![gray3](doc/gray3.png)

### gray with filter size 5 -> gray5.png

```console
./out/main ./doc/image.png ./doc/gray5.png 5
```

![gray5](doc/gray5.png)

### colorful with filter size 3 -> color3.png

```console
./out/main ./doc/image.png ./doc/colorful3.png 3 true
```

![color3](doc/colorful3.png)

### colorful with filter size 5 -> color5.png

```console
./out/main ./doc/image.png ./doc/colorful5.png 5 true
```

![color5](doc/colorful5.png)

