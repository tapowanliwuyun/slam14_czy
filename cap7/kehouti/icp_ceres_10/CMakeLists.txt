cmake_minimum_required(VERSION 2.8)

project(cap7_10_ceres_icp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)

# 添加cmake模块以使用ceres库
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 3.1 REQUIRED)
find_package(Ceres REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRECTORIES}
        "/usr/include/eigen3"
        ${CERES_INCLUDE_DIRECTORIES}
)

add_executable(cap7_10_ceres_icp src/icp_ceres.cpp)

target_link_libraries(cap7_10_ceres_icp
        ${OpenCV_LIBRARIES}
        ${CERES_LIBRARIES}
        )

