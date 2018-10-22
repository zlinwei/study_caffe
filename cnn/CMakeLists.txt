message(STATUS "${CMAKE_BINARY_DIR}")
message(STATUS "${CMAKE_SOURCE_DIR}")

file(GLOB PROTOTXT_FILES "${CMAKE_SOURCE_DIR}/cnn/*.prototxt")
message(STATUS ${PROTOTXT_FILES})
file(COPY ${PROTOTXT_FILES} DESTINATION ${CMAKE_BINARY_DIR}/cnn)

add_executable(cnn cnn.cpp)
target_link_libraries(cnn caffe caffeproto protobuf caffe glog gflags ${Boost_LIBRARIES})
