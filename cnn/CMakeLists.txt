message(STATUS "${CMAKE_BINARY_DIR}")
message(STATUS "${CMAKE_SOURCE_DIR}")

file(GLOB PROTOTXT_FILES "${CMAKE_SOURCE_DIR}/cnn/*.prototxt")
message(STATUS ${PROTOTXT_FILES})
file(COPY ${PROTOTXT_FILES} DESTINATION ${CMAKE_BINARY_DIR}/cnn)

add_library(DeepNet DeepNet.cpp)
target_link_libraries(DeepNet caffe caffeproto protobuf caffe glog gflags ${Boost_LIBRARIES})

add_executable(cnn cnn.cpp)
target_link_libraries(cnn DeepNet)
