# Loading Pytorch Model in C++
## Implementation of DNN face landmark detector from python by https://github.com/1adrianb/face-alignment
Please Refer []() for more details.
--
Prerequisite
* Visual studio 2015/17
* Libtorch 1.0.0
* dlib 19.16
* Opencv 2/3
--

./face-alignment/ folder contains the base python for the DNN implementation\
./cpp_app/ folder contains example of loading the torch script in C++

# Tracing Pytorch model for C++ API
1. To convert the pytorch network model for C++ use, the model must be traced.
There is two ways to convert the model into torch script.
1. Tracing
	* Fast, may not be able to handle complex control flow
2. Annotation
	* Slow (can be very slow), able to handle complex control flow
Please refer to Pytorch C++ documentation for more detail explanation

2. Serialize the model with it's weight
	* Save the model in file
3. Load it in C++ application
	* For the landmark detector, some pre-processing is done using dlib and pytorch.
	* In the C++ implementation, these are re-implemented using dlib for C++ and libtorch. (The result might not be 100% same with the python version)


## Including Libtorch (Pytorch C++) in visual studio
Using prebuild library for vs2015/17 x64 release only.
Please build from source for different build.
Include Directories
* $(ProjectDir)include\dlib
* $(ProjectDir)include\libtorch
* $(ProjectDir)include\libtorch\torch\csrc\api\include
* $(ProjectDir)include\opencv

Library Directories
* $(ProjectDir)lib\dlib
* $(ProjectDir)lib\libtorch
* $(ProjectDir)lib\opencv

Linker Input
* Add to vs project 'linker->input->additional dependencies'
* Copy the following
```
dlib19.16.0_release_64bit_msvc1900.lib
torch.lib
caffe2.lib
libprotobuf.lib
c10.lib
```

## Adding dlib source to visual studio
From vs solution explorer->properties->add existing item
Add '<current project folder>/include/dlib/dlib/all/source.cpp'

## Including all the required DLL files
Copy dll from x64->Release
```
torch.dll
caffe2.dll
libiomp5md.dll
c10.dll
(opencv).dll
```
