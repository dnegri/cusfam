cmake_minimum_required(VERSION 3.16)
project(cusfam CXX Fortran)
set(CMAKE_CXX_STANDARD 14)
add_compile_definitions(CPU)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY omp)
endif()

find_package(Ginkgo REQUIRED)
find_package(OpenMP REQUIRED)

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
        ../src/CSRSolver.*
        ../src/SuperLUSolver.*Î
        ../src/Feedback.*
        ../src/Simon.*
        ../src/myblas.h
        ../src/BICG*
        ../src/SteamTable.*
        ../src/DepletionChain.*
        ../src/SuperLUSolver.*
        ../src/GinkgoSolver.*
        )

include_directories(/opt/homebrew/include /usr/local/include/)
link_directories(/opt/homebrew/lib /usr/local/lib/)

SET_SOURCE_FILES_PROPERTIES(${SOURCES} PROPERTIES LANGUAGE CXX )
SET_SOURCE_FILES_PROPERTIES(../src/simondbreader.f90 PROPERTIES LANGUAGE Fortran )

add_library(L${PROJECT_NAME} ${SOURCES} )
add_executable(${PROJECT_NAME} ${SOURCES} ../src/simondbreader.f90)
target_link_libraries(${PROJECT_NAME} superlu blas Ginkgo::ginkgo OpenMP::OpenMP_CXX)

