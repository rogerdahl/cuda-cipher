cmake_minimum_required(VERSION 3.5)
project(cuda-cipher)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(SRC_OO_EXTRACT_DIR ${CMAKE_SOURCE_DIR}/src/oo_extract)
set(SRC_OO_CRACK ${CMAKE_SOURCE_DIR}/src/oo_crack)
set(SRC_CIPHER_LIB_DIR ${CMAKE_SOURCE_DIR}/src/cipher_lib)

SET(CUDA_VERBOSE_BUILD ON)
SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA QUIET REQUIRED)
message(STATUS "Found CUDA version: ${CUDA_VERSION}")
#set(BUILD_SHARED_LIBS OFF)

#set(Boost_USE_STATIC_LIBS   ON)
find_package(
  Boost 1.58.0 EXACT REQUIRED COMPONENTS
  filesystem
  program_options
  serialization
  system
  thread
)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(
  CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
  -gencode=arch=compute_61,code=sm_61
  -gencode arch=compute_50,code=sm_50
  -gencode arch=compute_35,code=sm_35
  -std=c++11;
)
# --device-c;-c;
# --relocatable-device-code=true

#set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "" )

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message(WARNING "Building DEBUG version")
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -g -G -O0)
else()
  message(WARNING "Building RELEASE version")
endif()

set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -Wall -Werror -Wfatal-errors -Wno-deprecated-declarations"
)

set(CMAKE_CXX_STANDARD 14)

# Specify include directories
include_directories(
  ${SRC_OO_EXTRACT_DIR}
  ${SRC_OO_CRACK}
  ${SRC_CIPHER_LIB_DIR}
  ${Boost_INCLUDE_DIRS}
)

link_directories(
  ${Boost_LIBRARY_DIRS}
)

link_libraries(
  ${Boost_LIBRARIES}
  ${CUDA_LIBRARIES}
)

#cuda_add_executable(
#  oo-extract
##  ${SRC_LIB_DIR}/cuda_util.cpp
##  ${SRC_OO_CRACK}/config.cpp
##  ${SRC_OO_CRACK}/cuda_timers.cpp
##  ${SRC_OO_CRACK}/kernels.cu
##  ${SRC_OO_CRACK}/main.cpp
##  ${SRC_OO_CRACK}/pch.cpp
##  ${SRC_OO_CRACK}/platform.cpp
##  ${SRC_OO_CRACK}/tracks.cpp
##  ${SRC_TRACK_DIR}/pch.cpp
##  ${SRC_TRACK_DIR}/track.cpp
#)

cuda_add_executable(
  oo-crack
  ${SRC_OO_CRACK}/cuda_crack_pw.cpp
  ${SRC_CIPHER_LIB_DIR}/check_passwords.cu
  ${SRC_CIPHER_LIB_DIR}/base64.cpp
  ${SRC_CIPHER_LIB_DIR}/cuda_utilities.cpp
  ${SRC_CIPHER_LIB_DIR}/passwords.cpp
  ${SRC_CIPHER_LIB_DIR}/rule_engine.cpp
)

message(WARNING "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")
message(WARNING "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
