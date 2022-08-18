cmake_minimum_required(VERSION 2.8)

project(orb_self)

set(CMAKE_CXX_FLAGS "-std=c++14 -mfma")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(orb_self src/orb_self.cpp)

target_link_libraries(orb_self ${OpenCV_LIBRARIES})
