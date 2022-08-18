cmake_minimum_required(VERSION 3.0)

project( stereoVision )

set(CMAKE_CXX_STANDARD 14)

include_directories("/usr/include/eigen3")
find_package(OpenCV 3 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}})
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(stereoVision src/stereoVision.cpp)

target_link_libraries(stereoVision ${OpenCV_LIBRARIES} ${Pangolin_LIBRARIES})