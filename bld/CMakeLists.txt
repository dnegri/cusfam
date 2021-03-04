cmake_minimum_required(VERSION 3.16)
project(cusfam CXX Fortran)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY omp)
endif()

find_package(Ginkgo REQUIRED)
find_package(OpenMP REQUIRED)

file(GLOB SOURCES
        ../src/pch.h
        ../src/CrossSection.*
        ../src/Geometry.*
        ../src/ControlRod.*
        ../src/main.cpp
        ../src/sanm2n.h
        ../src/sanm2n.cpp
        ../src/Nodal.*
        ../src/NodalCPU.*
        ../src/mat2g.h
        ../src/CMFD.*
        ../src/CSRSolver.*
#        ../src/SuperLUSolver.*
        ../src/Feedback.*
        ../src/myblas.h
        ../src/BICGCMFD.*
        ../src/BICGSolver.*
        ../src/SteamTable.*
        ../src/Depletion.*
        ../src/GinkgoSolver.*
        ../src/JacobiBicgSolver.*
        ../src/Simon.*
        ../src/SimonCPU.*
        )

file(GLOB FortranSources
        ../src/CIsotope.f90
        ../src/CReflector.f90
        ../src/CComposition.f90
        ../src/CTableSet.f90
        ../src/simondbreader.f90
        ../src/cusfam.f90
        ../src/CSteamTable.f90
        ../src/quad1.f90
        ../src.nist/*.f)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_Fortran_FLAGS "-ffree-line-length-none -finit-local-zero -fno-automatic -fdefault-real-8")
set(CMAKE_CXX_FLAGS "-ffpe-trap=invalid,zero,overflow")

include_directories(/usr/local/include /opt/homebrew/include)
link_directories(/usr/local/lib /opt/homebrew/lib)

SET_SOURCE_FILES_PROPERTIES(${SOURCES} PROPERTIES LANGUAGE CXX )
SET_SOURCE_FILES_PROPERTIES(${FortranSources} PROPERTIES LANGUAGE Fortran )

add_executable(${PROJECT_NAME} ${FortranSources} ${SOURCES} )
target_link_libraries(${PROJECT_NAME} Ginkgo::ginkgo OpenMP::OpenMP_CXX)

