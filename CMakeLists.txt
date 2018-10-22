cmake_minimum_required(VERSION 3.12)
project(study_caffe)

set(CMAKE_CXX_STANDARD 14)

#include_directories(/usr/local/cuda/include)
#link_directories(/usr/local/cuda/lib64)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -O3")

#add_definitions(-DCPU_ONLY)

include_directories(/usr/local/opt/lapack/include)
link_directories(/usr/local/opt/lapack/lib)

include_directories(/usr/local/opt/openblas/include)
link_directories(/usr/local/opt/openblas/lib)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib)

# Boost
set(Boost_USE_STATIC_LIBS ON)  # only find static libs
set(Boost_USE_DEBUG_LIBS OFF) # ignore debug libs and
set(Boost_USE_RELEASE_LIBS ON)  # only find release libs
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost COMPONENTS system program_options thread timer chrono REQUIRED)
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
endif ()

message(STATUS ${Boost_LIBRARIES})

add_subdirectory(dnn)


enable_testing()
add_subdirectory(tests)
