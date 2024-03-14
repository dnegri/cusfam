cmake_minimum_required(VERSION 3.16)
project(cusfam C CXX Fortran)

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(OpenMP REQUIRED)
find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(fmt REQUIRED)
find_package(plog REQUIRED)

include_directories(../../dnegri.jiarray/src)
add_subdirectory(../../nist ${CMAKE_CURRENT_BINARY_DIR}/nist)

file(
  GLOB
  SOURCES
  ./src/CrossSection.cpp
  ./src/Geometry.cpp
  ./src/ControlRod.cpp
  ./src/main.cpp
  ./src/sanm2n.cpp
  ./src/Nodal.cpp
  ./src/NodalCPU.cpp
  ./src/CMFD.cpp
  ./src/CSRSolver.cpp
  ./src/Feedback.cpp
  ./src/BICGCMFD.cpp
  ./src/BICGSolver.cpp
  ./src/SteamTable.cpp
  ./src/Depletion.cpp
  ./src/GinkgoSolver.cpp
  ./src/JacobiBicgSolver.cpp
  ./src/LineJacobiBicgSolver.cpp
  ./src/SSORSolver.cpp
  ./src/Simon.cpp
  ./src/SimonCPU.cpp
  ./src/PinPower.cpp
  ./src/ShapeMatch.cpp
  ./src/ASISearcher.cpp)

file(
  GLOB
  FortranSources
  ./src/CIsotope.f90
  ./src/CReflector.f90
  ./src/CComposition.f90
  ./src/CTableSet.f90
  ./src/CFormFunction.f90
  ./src/simondbreader.f90
  # ./src/cusfam.f90
  ./src/CSteamTable.f90
  ./src/quad1.f90)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -DJIARRAY_OFFSET=0 -fopenmp=libgomp ")
set(CMAKE_CXX_FLAGS_DEBUG
    " ${CMAKE_CXX_FLAGS_DEBUG} -DJIARRAY_OFFSET=0 -DJIARRAY_DEBUG -fsanitize=address"
)
set(CMAKE_Fortran_FLAGS
    "${CMAKE_Fortran_FLAGS} -ffree-line-length-none -finit-local-zero -frecursive -fdefault-real-8 -fdefault-double-8 -cpp -DXS_PREC=8"
)

set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${FortranSources} PROPERTIES LANGUAGE Fortran)

# simon test executable
add_executable(${PROJECT_NAME}.main ${src.nist} ${FortranSources} ${SOURCES})
target_link_libraries(
  ${PROJECT_NAME}.main PRIVATE Python3::Python fmt::fmt-header-only plog::plog
                               nist OpenMP::OpenMP_CXX)

# pybind11 method:
pybind11_add_module(${PROJECT_NAME} ${FortranSources} ${SOURCES}
                    ./src/python_module.cpp)
target_link_libraries(
  ${PROJECT_NAME} PRIVATE Python3::Python fmt::fmt-header-only plog::plog
                          nist OpenMP::OpenMP_CXX)

add_custom_command(
  TARGET ${PROJECT_NAME}
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${PROJECT_NAME}>
          ${CMAKE_CURRENT_SOURCE_DIR}/python)
