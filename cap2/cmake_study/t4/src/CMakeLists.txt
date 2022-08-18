ADD_EXECUTABLE(main main.c)

#使用环境变量
set(CMAKE_INCLUDE_PATH /tmp/t3/usr/include/hello )
FIND_PATH(myHeader hello.h)
IF(myHeader)
INCLUDE_DIRECTORIES(${myHeader})
ENDIF(myHeader)
# 添加头文件寻找的路径
#INCLUDE_DIRECTORIES(/tmp/t3/usr/include/hello)
#两种方法用一种


# 对指定路径寻找hello库，并保存到变量my_hello，此时是动态库
#这句话的意思是：在/tmp/t3/usr/lib路径下，搜索hello库，并保存到my_hello变量中
# FIND_LIBRARY(my_hello hello /tmp/t3/usr/lib)

# 对指定路径寻找libhello.a库，并保存到变量my_hello，此时是静态库
FIND_LIBRARY(my_hello libhello.a /tmp/t3/usr/lib)

#静态库和共享库用一种即可

TARGET_LINK_LIBRARIES(main ${my_hello})
MESSAGE(STATUS "HOME dir：" $ENV{HOME})
MESSAGE(${CMAKE_MAJOR_VERSION})
MESSAGE(${CMAKE_MINOR_VERSION})
MESSAGE(${CMAKE_PATCH_VERSION})
MESSAGE(${CMAKE_SYSTEM})
MESSAGE(${CMAKE_SYSTEM_NAME})
MESSAGE(${CMAKE_SYSTEM_VERSION})
MESSAGE(${CMAKE_SYSTEM_PROCESSOR})
MESSAGE(${CMAKE_SYSTEM_VERSION})
MESSAGE(${UNIX})
