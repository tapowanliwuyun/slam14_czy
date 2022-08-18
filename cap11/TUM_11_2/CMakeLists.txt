cmake_minimum_required(VERSION 2.8)

project(tum)

include_directories("/usr/include/eigen3")
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(tum src/tum.cpp)
target_link_libraries(tum ${Pangolin_LIBRARIES})

