cmake_minimum_required(VERSION 2.8)
project(pose_graph_g2o_SE3)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# Eigen
include_directories("/usr/include/eigen3")

# sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# g2o
find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

SET(G2O_LIBS  g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension g2o_types_slam3d cxsparse)

add_executable(pose_graph_g2o_SE3 src/pose_graph_g2o_SE3.cpp)
target_link_libraries(pose_graph_g2o_SE3
        ${G2O_LIBS}
        ${CHOLMOD_LIBRARIES} fmt
        )
