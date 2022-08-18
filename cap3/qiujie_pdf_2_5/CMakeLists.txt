CMAKE_MINIMUM_REQUIRED( VERSION 2.8)
PROJECT( qiujie )

set( CMAKE_BUILD_TYPE "Release" )

INCLUDE_DIRECTORIES( "/usr/include/eigen3" )

ADD_EXECUTABLE( qiujie src/qiujie.cpp )