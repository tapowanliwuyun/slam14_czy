cmake_minimum_required( VERSION 2.8 )
project( joinMap )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_CXX_STANDARD 11 )#or 4

# opencv
find_package( OpenCV 3 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

# eigen
include_directories("/usr/include/eigen3")

# pcl
#find_package( PCL REQUIRED COMPONENT common io)
find_package( PCL 1.8 REQUIRED)
set(PCL_INCLUDE_DIRS /usr/include/pcl-1.8)  #指定pcl1.8路径
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )

add_executable( joinMap src/joinMap.cpp )
target_link_libraries( joinMap ${OpenCV_LIBS} ${PCL_LIBRARIES} )