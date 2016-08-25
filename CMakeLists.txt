cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(model_extraction)

set(CMAKE_BUILD_TYPE Debug)

find_package(PCL 1.8 REQUIRED)
find_package(Boost 1.50 COMPONENTS filesystem REQUIRED)

include_directories(${Boost_INCLUDE_DIR})
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(model_extraction model_extraction.cpp)
target_link_libraries(model_extraction ${PCL_LIBRARIES})
target_link_libraries(model_extraction ${Boost_LIBRARIES})