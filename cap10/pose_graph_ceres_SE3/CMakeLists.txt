cmake_minimum_required(VERSION 2.8)

project(pose_graph_ceres_SE3)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)

include_directories("/usr/include/eigen3")
find_package(Sophus REQUIRED)
Find_Package(Ceres REQUIRED)

set(Sophus_LIBRARIES "/usr/local/lib/libSophus.so")

include_directories(${Sophus_INCLUDE_DIRS} )

add_executable(pose_graph_ceres_SE3 src/pose_graph_ceres_SE3.cpp)

target_link_libraries(pose_graph_ceres_SE3 ${CERES_LIBRARIES} ${Sophus_LIBRARIES} fmt)
