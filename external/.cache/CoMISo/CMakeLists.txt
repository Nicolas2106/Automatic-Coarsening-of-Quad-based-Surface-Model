# Distributed under the OSI-approved MIT License.  See accompanying
# file LICENSE or https://github.com/Crascit/DownloadProject for details.

cmake_minimum_required(VERSION 3.1)

project(CoMISo-download NONE)

include(ExternalProject)
ExternalProject_Add(CoMISo-download
                    TLS_VERIFY;OFF;GIT_CONFIG;advice.detachedHead=false;GIT_REPOSITORY;https://github.com/libigl/CoMISo.git;GIT_TAG;d60aa4759fba76b0b793b1efb090b7a771dd7c56
                    SOURCE_DIR          "F:/Users/Nicolas/Desktop/TESI/Quadrilateral extension to libigl/libigl/cmake/../external/CoMISo"
                    BINARY_DIR          "F:/Users/Nicolas/Desktop/TESI/Quadrilateral extension to libigl/libigl/build/CoMISo-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
)
