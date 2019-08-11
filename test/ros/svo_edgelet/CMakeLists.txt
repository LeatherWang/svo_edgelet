cmake_minimum_required(VERSION 2.8.3)
project(svo_ros_leather)

include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

rosbuild_init()

IF(NOT ROS_BUILD_TYPE)
  SET(ROS_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${ROS_BUILD_TYPE})


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall  -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()


#find_package(catkin REQUIRED COMPONENTS
#  roscpp
#  rospy
#  std_msgs
#  message_generation
#  tf
#  cv_bridge
#  message_filters
#)

find_package(OpenCV REQUIRED)
find_package(Pangolin REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS thread system filesystem)

#catkin_package(
#  CATKIN_DEPENDS message_runtime
#)

include_directories(
  include
#  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  "/usr/include/eigen3"
  ${Pangolin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/../../../include
)

set(LIBS
#  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
  ${Pangolin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${PROJECT_SOURCE_DIR}/../../../lib/libsvo.so
)

#ADD_EXECUTABLE(svo_ros_live ./svo_ros_live.cpp)
rosbuild_add_executable(svo_ros_live ./svo_ros_live.cpp)
TARGET_LINK_LIBRARIES(svo_ros_live ${LIBS}
)





