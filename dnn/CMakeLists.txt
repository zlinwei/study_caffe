message(STATUS "${CMAKE_BINARY_DIR}")
message(STATUS "${CMAKE_SOURCE_DIR}")

file(GLOB PROTOTXT_FILES "${CMAKE_SOURCE_DIR}/dnn/*.prototxt")
message(STATUS ${PROTOTXT_FILES})
file(COPY ${PROTOTXT_FILES} DESTINATION ${CMAKE_BINARY_DIR}/dnn)

add_executable(dnn dnn.cpp DenseNet.cpp)
target_link_libraries(dnn caffe caffeproto protobuf caffe glog gflags ${Boost_LIBRARIES})