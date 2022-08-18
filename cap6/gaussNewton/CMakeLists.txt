cmake_minimum_required(VERSION 2.8)

project(gaussNewton)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(OpenCV 3 REQUIRED)

include_directories("/usr/include/eigen3")
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(gaussNewton src/gaussNewton.cpp)

target_link_libraries(gaussNewton ${OpenCV_LIBRARIES})
