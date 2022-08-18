cmake_minimum_required(VERSION 2.8)

project(all_paras_g2o)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
include_directories(
        "/usr/include/eigen3"
)

Find_Package(G2O REQUIRED)
Find_Package(Eigen3 REQUIRED)
Find_Package(Ceres REQUIRED)
Find_Package(Sophus REQUIRED)
Find_Package(CSparse REQUIRED)

SET(G2O_LIBS  g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension g2o_types_slam3d cxsparse)

include_directories(${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR}
        ${G2O_INCLUDE_DIRECTORIES}
        ${Sophus_INCLUDE_DIRECTORIES}
        )

add_library(bal_common src/common.cpp)
add_executable(all_paras_g2o src/all_paras_g2o.cpp)

target_link_libraries(all_paras_g2o bal_common
        ${G2O_LIBS}
        ${CSPARSE_LIBRARY}
        ${Sophus_LIBRARIES} fmt
        )

