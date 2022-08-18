cmake_minimum_required(VERSION 2.8)
project(imageBasics)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package( OpenCV 3 REQUIRED )

include_directories(${OpenCV_INCLUDE_DIR})
add_executable(imageBasics src/imageBasics.cpp)

target_link_libraries( imageBasics ${OpenCV_LIBS} )
