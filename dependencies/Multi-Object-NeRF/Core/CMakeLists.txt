# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Modification
# @author Xiao Han
#

cmake_minimum_required(VERSION 3.18)

#PROJECT(instant-ngp
#	VERSION 1.0
#	DESCRIPTION "Instant neural graphics primitives"
#	LANGUAGES C CXX CUDA
#)
PROJECT(Multi_Object_NeRF
	VERSION 1.0
	DESCRIPTION "Multi Object NeRF"
	LANGUAGES C CXX CUDA
)

#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo)
ENDIF()

set(NGP_VERSION "${CMAKE_PROJECT_VERSION}")

if (NOT NGP_DEPLOY)
	set(NGP_VERSION "${NGP_VERSION}dev")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

###############################################################################
# Build type and C++ compiler setup
###############################################################################

# Set a default configuration if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "No release type specified. Setting to 'Release'.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-deprecated")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

###############################################################################
# CUDA compiler setup
###############################################################################

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)


list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-mf16c")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-Wno-float-conversion")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fno-strict-aliasing")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fPIC")
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")

list(APPEND CUDA_NVCC_FLAGS "-DCUDA_API_PER_THREAD_DEFAULT_STREAM=1")

###############################################################################
# Dependencies
###############################################################################

################
# tiny-cuda-nn #
################
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory(third_party/tiny-cuda-nn)

set(CMAKE_CUDA_ARCHITECTURES ${TCNN_CUDA_ARCHITECTURES})

include_directories("../third_party/tiny-cuda-nn/include")
include_directories("../third_party/tiny-cuda-nn/dependencies")
include_directories("../third_party")

#########
# Eigen #
#########
find_package(Eigen3 REQUIRED)

##########
# OpenCV #
##########
find_package(OpenCV 3.0 REQUIRED)

###############################################################################
# Program
###############################################################################

list(APPEND NGP_DEFINITIONS -DNGP_VERSION="${NGP_VERSION}")
list(APPEND NGP_LIBRARIES ${CMAKE_DL_LIBS})

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
)

set(SOURCES
	src/nerf.cu
	src/nerf_model.cu
	src/nerf_data.cu
	src/marching_cubes.cu
	src/nerf_manager.cu
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR})

get_filename_component(CUDA_COMPILER_BIN "${CMAKE_CUDA_COMPILER}" DIRECTORY)


add_library(MON SHARED ${SOURCES})

set_target_properties(MON PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(MON PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(MON PUBLIC ${NGP_DEFINITIONS})
target_compile_options(MON PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
target_link_libraries(MON PUBLIC ${NGP_LIBRARIES} ${OpenCV_LIBS} tiny-cuda-nn -lcurand)

