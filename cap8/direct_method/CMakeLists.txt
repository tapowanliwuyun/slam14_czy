cmake_minimum_required(VERSION 2.8)
project(direct_method)

set(CMAKE_BUILD_TYPE "Release")
add_definitions("-DENABLE_SSE")
set(CMAKE_CXX_STANDARD 14 )

find_package(OpenCV 3 REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        "/usr/include/eigen3/"
        ${Pangolin_INCLUDE_DIRS}
)

add_executable(direct_method src/direct_method.cpp)
target_link_libraries(direct_method
        ${OpenCV_LIBS}
        ${Pangolin_LIBRARIES}
        ${Sophus_LIBRARIES} fmt
        )



