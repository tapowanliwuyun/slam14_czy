cmake_minimum_required(VERSION 2.8)
project(pose_estimation_2d2d)

set(CMAKE_BUILD_TYPE "Release")
set( CMAKE_CXX_STANDARD 14)

# 添加cmake模块以使用g2o
#list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.1 REQUIRED)
# find_package( OpenCV REQUIRED ) # use this if in OpenCV2
include_directories("/usr/include/eigen3")
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(pose_estimation_2d2d src/pose_estimation_2d2d.cpp)

target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBRARIES})

