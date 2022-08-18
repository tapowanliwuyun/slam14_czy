cmake_minimum_required(VERSION 2.8)

project(orb_cv)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE "Release")

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(orb_cv src/orb_cv.cpp)

target_link_libraries(orb_cv ${OpenCV_LIBRARIES})

