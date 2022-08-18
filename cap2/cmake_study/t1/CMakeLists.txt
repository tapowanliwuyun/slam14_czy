project(HELLO)
set(SRC_LIST src/main.cpp)
message(STATUS "This is BINARY dir" ${HELLO_BINARY_DIR})
message(STATUS "This is SOURCE dir" ${HELLO_SOURCE_DIR})
add_executable(hello ${SRC_LIST})


