# Automatic Coarsening of Quad-based Surface Model

A university project for a Master's degree thesis in Computer Science at the University of Milan.

## Dependencies

The only dependencies are STL, Eigen, [libigl](http://libigl.github.io/libigl/) and the dependencies
of the `igl::opengl::glfw::Viewer` (OpenGL, glad and GLFW).
The CMake build system will automatically download libigl and its dependencies using
[CMake FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html),
thus requiring no setup on your part.

To use a local copy of libigl rather than downloading the repository via FetchContent, you can use
the CMake cache variable `FETCHCONTENT_SOURCE_DIR_LIBIGL` when configuring your CMake project for
the first time:
```
cmake -DFETCHCONTENT_SOURCE_DIR_LIBIGL=<path-to-libigl> ..
```
When changing this value, do not forget to clear your `CMakeCache.txt`, or to update the cache variable
via `cmake-gui` or `ccmake`.

## Compile

Compile this project using the standard cmake routine:

    mkdir build
    cd build
    cmake ..
    make

This should find and build the dependencies.

## Run

From within the `build` directory just find the file `libigl.sln` and open it with Visual Studio.

## Results
<img src="https://github.com/Nicolas2106/Automatic-Coarsening-of-Quad-based-Surface-Model/blob/main/figures/teddybear.png" width="70%">
<img src="https://github.com/Nicolas2106/Automatic-Coarsening-of-Quad-based-Surface-Model/blob/main/figures/david.png" width="70%">
<img src="https://github.com/Nicolas2106/Automatic-Coarsening-of-Quad-based-Surface-Model/blob/main/figures/bimba.png" width="70%">
<img src="https://github.com/Nicolas2106/Automatic-Coarsening-of-Quad-based-Surface-Model/blob/main/figures/julius.png" width="70%">
<img src="https://github.com/Nicolas2106/Automatic-Coarsening-of-Quad-based-Surface-Model/blob/main/figures/screw.png" width="65%">
<img src="https://github.com/Nicolas2106/Automatic-Coarsening-of-Quad-based-Surface-Model/blob/main/figures/molecule.png" width="70%">
<img src="https://github.com/Nicolas2106/Automatic-Coarsening-of-Quad-based-Surface-Model/blob/main/figures/armadillo.png" width="58%">

### Extreme coarsening
<img src="https://github.com/Nicolas2106/Automatic-Coarsening-of-Quad-based-Surface-Model/blob/main/figures/david2.png" width="60%">
<img src="https://github.com/Nicolas2106/Automatic-Coarsening-of-Quad-based-Surface-Model/blob/main/figures/knot.png" width="47%">

### Surface quality improvement
<img src="https://github.com/Nicolas2106/Automatic-Coarsening-of-Quad-based-Surface-Model/blob/main/figures/teddybear2.png" width="70%">
<img src="https://github.com/Nicolas2106/Automatic-Coarsening-of-Quad-based-Surface-Model/blob/main/figures/tangle.png" width="70%">
