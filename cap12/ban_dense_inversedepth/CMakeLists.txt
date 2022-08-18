cmake_minimum_required(VERSION 2.8)
project(ban_dense_inversedepth)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)

#sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
set(Sophus_LIBRARIES "/usr/local/lib/libSophus.so")

# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# eigen
include_directories("/usr/include/eigen3/")

# pcl
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(ban_dense_inversedepth src/ban_dense_inversedepth.cpp)
target_link_libraries(ban_dense_inversedepth ${OpenCV_LIBS} ${PCL_LIBRARIES} ${Sophus_LIBRARIES} fmt)

