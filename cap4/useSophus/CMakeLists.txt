cmake_minimum_required(VERSION 2.8)
project(useSophus)

include_directories("/usr/include/eigen3")

# 为使用 sophus，需要使用find_package命令找到它
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

# Eigen

add_executable(useSophus src/useSophus.cpp)
target_link_libraries(useSophus ${Sophus_LIBRARIES} fmt)


