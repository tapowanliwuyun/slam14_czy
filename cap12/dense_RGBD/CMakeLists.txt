cmake_minimum_required(VERSION 2.8)
project(dense_RGBD)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)

# opencv
find_package(OpenCV 3.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# eigen
include_directories("/usr/include/eigen3/")

# pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

# octomap
#find_package(octomap REQUIRED)
#include_directories(${OCTOMAP_INCLUDE_DIRS})

add_executable(pointcloud_mapping src/pointcloud_mapping.cpp)
target_link_libraries(pointcloud_mapping ${OpenCV_LIBS} ${PCL_LIBRARIES})

