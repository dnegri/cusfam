cmake_minimum_required(VERSION 3.16)
project(cusfam)
set(CMAKE_CXX_STANDARD 14)
add_compile_definitions(CPU BZ_DEBUG)

file(GLOB SOURCES
        ../src/pch.h
        ../src/CrossSection.h
        ../src/Solution.cpp
        ../src/Solution.h
        ../src/CrossSection.cpp
        ../src/Geometry.cpp
        ../src/Geometry.h
        ../src/main.cpp
        ../src/sanm2n.h
        ../src/sanm2n.cpp
        ../src/Nodal.*
        ../src/NodalCPU.*
        ../src/mat2g.h
        ../src/CMFD*
#        ../src/BICG*
        )
file(GLOB eigen ../src/eigen/*)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

SET_SOURCE_FILES_PROPERTIES(${SOURCES} ${eigen} PROPERTIES LANGUAGE CXX )
 
add_library(L${PROJECT_NAME} ${SOURCES} )
add_executable(${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} blitz)

