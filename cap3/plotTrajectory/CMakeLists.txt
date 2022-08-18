cmake_minimum_required(VERSION 2.8)

project(plotTrajectory)

include_directories("/usr/include/eigen3")
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(plottrajectory src/plotTrajectory.cpp)
target_link_libraries(plottrajectory ${Pangolin_LIBRARIES})
