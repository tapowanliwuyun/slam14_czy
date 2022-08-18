cmake_minimum_required(VERSION 2.8)

project( undistortImage )

set(CMAKE_CXX_FLAGS "-std=c++11")
find_package( OpenCV 3 REQUIRED )

include_directories(${OpenCV_INCLUDE_DIR})

add_executable(undistortImage src/undistortImage.cpp)

target_link_libraries(undistortImage ${OpenCV_LIBS})