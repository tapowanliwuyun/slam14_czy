cmake_minimum_required(VERSION 3.4.3)
project(HELLO)

FOREACH(A RANGE 5 15 3)
MESSAGE(${A})
ENDFOREACH(A)

add_subdirectory(src bin)

ADD_TEST(mytest ${PROJECT_BINARY_DIR}/bin/hello)
ENABLE_TESTING()

#INSTALL(FILES COPYRIGHT README DESTINATION share/doc/cmake/t2)
#INSTALL(PROGRAMS runhello.sh DESTINATION bin)
#INSTALL(DIRECTORY doc/ DESTINATION share/doc/cmake/t2)


