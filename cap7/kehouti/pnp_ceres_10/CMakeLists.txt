cmake_minimum_required(VERSION 2.8)
project(cap7_10_ceres_pnp)

set( CMAKE_CXX_STANDARD 14)
set( CMAKE_BUILD_TYPE Release )

# 添加cmake模块以使用ceres库
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package(OpenCV 3.1 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRECTORIES})

# 寻找Ceres库并添加它的头文件
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

include_directories("/usr/include/eigen3")

add_executable(cap7_10_ceres_pnp src/pnp_ceres.cpp)

target_link_libraries( cap7_10_ceres_pnp
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        )
