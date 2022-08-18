cmake_minimum_required(VERSION 2.8)

project(useLK)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV 3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRECTORIES})

add_executable(useLK src/useLK.cpp)

target_link_libraries(useLK ${OpenCV_LIBRARIES})
