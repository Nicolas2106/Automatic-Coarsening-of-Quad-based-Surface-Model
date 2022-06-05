cmake_minimum_required(VERSION 3.1)
project(libigl_tutorials)

### Conditionally compile certain modules depending on libraries found on the system
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)

### Compile libigl in header-only mode for Python bindings
option(LIBIGL_USE_STATIC_LIBRARY "Use LibIGL as static library" ON)

### Adding libIGL: choose the path to your local copy libIGL
if(NOT TARGET igl_common)
  include(libigl)
else()
  include(LibiglDownloadExternal)
endif()

### Download data
igl_download_tutorial_data()

### Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

### Choose which chapters to compile
option(APP "Compile chapter 1" ON)

# Store location of tutorial/shared directory
set(TUTORIAL_SHARED_PATH ${LIBIGL_EXTERNAL}/../tutorial/data CACHE PATH "location of shared tutorial resources")
add_library(application INTERFACE)
target_compile_definitions(application INTERFACE "-DTUTORIAL_SHARED_PATH=\"${TUTORIAL_SHARED_PATH}\"")
target_include_directories(application INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

#######################
if(NOT (LIBIGL_WITH_OPENGL AND LIBIGL_WITH_OPENGL_GLFW) )
  message(WARNING "Most executables depend on OpenGL and glfw. Use `cmake ../ -DLIBIGL_WITH_OPENGL=ON -DLIBIGL_WITH_OPENGL_GLFW=ON`")
  return()
endif()

if(APP)
  add_subdirectory("src")
endif()
