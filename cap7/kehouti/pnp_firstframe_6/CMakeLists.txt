cmake_minimum_required(VERSION 2.8)
project(cap7_6_pnp)

set( CMAKE_CXX_STANDARD 14)
#SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )
set( CMAKE_BUILD_TYPE Release )
# 添加cmake模块以使用g2o
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

find_package(OpenCV 3 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRECTORIES})

include_directories("/usr/include/eigen3")

find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRECTORIES})

find_package(CSparse REQUIRED)
include_directories(${CSPARSE_INCLUDE_DIR})

find_package(Sophus  REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(cap7_6_pnp src/pnp_firstframe.cpp)

target_link_libraries( cap7_6_pnp
        ${OpenCV_LIBS}
        g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
        g2o_types_slam3d
        #${G2O_LIBS}
        ${CSPARSE_LIBRARY}
        ${Sophus_LIBRARIES} fmt
        )
