cmake_minimum_required(VERSION 3.18)
project(RO-MAP)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()


add_definitions(-Wno-unused-value -Wno-unknown-pragmas -Wno-sequence-point 
                    -Wno-delete-non-virtual-dtor -Wno-unused-but-set-variable 
                    -Wno-sign-compare -Wno-unused-variable -Wno-return-local-addr 
                    -Wno-unused-function -Wno-deprecated-declarations)


MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

##########
# OpenCV #
##########
find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
   find_package(OpenCV 2.4.3 QUIET)
   if(NOT OpenCV_FOUND)
      message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
   endif()
endif()

#########
# Eigen #
#########
find_package(Eigen3 REQUIRED)

############
# Pangolin #
############
find_package(Pangolin REQUIRED)

##################
# Mutil_Object_NeRF #
##################
include_directories(${PROJECT_SOURCE_DIR}/dependencies/Multi-Object-NeRF/Core)
include_directories(${PROJECT_SOURCE_DIR}/dependencies/Multi-Object-NeRF/Core/include)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
#OBJECT-NERF-SLAM
src/ObjectFrame.cc
src/OptimizeObject.cc
src/ObjectMap.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/dependencies/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/dependencies/g2o/lib/libg2o.so
${PROJECT_SOURCE_DIR}/dependencies/line_lbd/lib/libline_lbd.so
#Multi-Object-NeRF
${PROJECT_SOURCE_DIR}/dependencies/Multi-Object-NeRF/Core/build/libMON.so
)

# Build examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/examples/Monocular)
add_executable(mono_tum
examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})
