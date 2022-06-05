# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 3.1)

project(catch2-download NONE)

include(ExternalProject)
ExternalProject_Add(catch2-download
                    TLS_VERIFY;OFF;GIT_CONFIG;advice.detachedHead=false;GIT_REPOSITORY;https://github.com/catchorg/Catch2.git;GIT_TAG;v2.11.0
                    SOURCE_DIR          "F:/Users/Nicolas/Desktop/TESI/Quadrilateral extension to libigl/libigl/cmake/../external/catch2"
                    BINARY_DIR          "F:/Users/Nicolas/Desktop/TESI/Quadrilateral extension to libigl/libigl/build/catch2-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
