cmake_minimum_required(VERSION 2.8)
project(pnps)

set( CMAKE_CXX_STANDARD 14)
set( CMAKE_BUILD_TYPE Release )

find_package(OpenCV 3.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRECTORIES})

include_directories("/usr/include/eigen3")

add_executable(pnps src/pnps.cpp)

target_link_libraries( pnps
        ${OpenCV_LIBS}
        )

