# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 3.1)

project(embree-download NONE)

include(ExternalProject)
ExternalProject_Add(embree-download
                    TLS_VERIFY;OFF;GIT_CONFIG;advice.detachedHead=false;GIT_REPOSITORY;https://github.com/embree/embree.git;GIT_TAG;v3.12.1
                    SOURCE_DIR          "F:/Users/Nicolas/Desktop/TESI/Quadrilateral extension to libigl/libigl/cmake/../external/embree"
                    BINARY_DIR          "F:/Users/Nicolas/Desktop/TESI/Quadrilateral extension to libigl/libigl/build/embree-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
