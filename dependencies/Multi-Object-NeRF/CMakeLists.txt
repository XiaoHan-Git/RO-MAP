cmake_minimum_required(VERSION 3.18)

PROJECT(OfflineNeRF LANGUAGES CXX)

#SET(CMAKE_BUILD_TYPE "Debug")
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall") 

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE RelWithDebInfo)
ENDIF()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-deprecated")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

####################
# Mutil_Object_NeRF#
####################
#add_subdirectory("Core")
include_directories("Core")
include_directories("Core/include")

#########
# Eigen #
#########
find_package(Eigen3 REQUIRED)

############
# Pangolin #
############
find_package(Pangolin REQUIRED)

##########
# OpenCV #
##########
find_package(OpenCV 3.0 REQUIRED)

include_directories(
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

add_executable(OfflineNeRF main.cpp)
target_link_libraries(OfflineNeRF 
${Pangolin_LIBRARIES} 
${OpenCV_LIBS}
${PROJECT_SOURCE_DIR}/Core/build/libMON.so
-lpthread
)
