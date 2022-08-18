cmake_minimum_required(VERSION 2.8)

project(pose_estimation_3d3d)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release )

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# eigen3
include_directories("/usr/include/eigen3")

#opencv2
find_package(OpenCV 3.1 REQUIRED)

#g2o
find_package(G2O REQUIRED)
find_package(CSparse REQUIRED)

#Sophus
find_package(Sophus REQUIRED)


include_directories(
        ${OpenCV_INCLUDE_DIRECTORIES}
        ${G2O_INCLUDE_DIRECTORIES}
        ${CSPARSE_INCLUDE_DIR}
        ${Sophus_INCLUDE_DIRECTORIES}

)

add_executable(pose_estimation_3d3d src/pose_estimation_3d3d.cpp)

target_link_libraries(pose_estimation_3d3d
        ${OpenCV_LIBRARIES}
        g2o_core g2o_stuff g2o_types_sba
        g2o_csparse_extension
        g2o_types_slam3d
        ${CSPARSE_LIBRARY}
        ${Sophus_LIBRARIES} fmt
        )
