SET(LIBHELLO_SRC hello.c)
#创建静态库
ADD_LIBRARY(hello SHARED ${LIBHELLO_SRC})
#创建共享库
ADD_LIBRARY(hello_static STATIC ${LIBHELLO_SRC})
#确保共享库和静态库名称相同
SET_TARGET_PROPERTIES(hello_static PROPERTIES OUTPUT_NAME "hello")
#输出共享库的名称修改输出信息
GET_TARGET_PROPERTY(OUTPUT_VALUE hello_static OUTPUT_NAME)
MESSAGE(STATUS "This is the hello_static OUTPUT_NAME:"${OUTPUT_VALUE})
# 避免在构建`libhello.a`时，就会清理掉`libhello.so`
SET_TARGET_PROPERTIES(hello PROPERTIES CLEAN_DIRECT_OUTPUT 1)
SET_TARGET_PROPERTIES(hello_static PROPERTIES CLEAN_DIRECT_OUTPUT 1)
#动态库版本号，动态库版本为1.2 API版本为1
SET_TARGET_PROPERTIES(hello PROPERTIES VERSION 1.2 SOVERSION 1)

#但是这里我把cmakelists.txt中的安装代码都注释掉了，避免对系统的干扰
# 安装到/tmp/t3/usr,即cmake -DCMAKE_INSTALL_PREFIX=/tmp/t3/usr ..
#INSTALL(TARGETS hello hello_static
#		LIBRARY DESTINATION lib
#		ARCHIVE DESTINATION lib)
#INSTALL(FILES hello.h DESTINATION include/hello)
