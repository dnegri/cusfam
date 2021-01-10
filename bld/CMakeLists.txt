cmake_minimum_required(VERSION 3.16)
project(cusfam)
set(CMAKE_CXX_STANDARD 14)

file(GLOB SOURCES
        ../src/pch.h
        ../src/CrossSection.h
        ../src/Solution.cpp
        ../src/Solution.h
        ../src/CrossSection.cpp
        ../src/Geometry.cpp
        ../src/Geometry.h
        ../src/main.cpp)

SET_SOURCE_FILES_PROPERTIES(${SOURCES} PROPERTIES LANGUAGE CXX )
 
add_library(L${PROJECT_NAME} ${SOURCES} )
add_executable(${PROJECT_NAME} ${SOURCES} )

