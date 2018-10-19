message(STATUS "${CMAKE_BINARY_DIR}")
message(STATUS "${CMAKE_SOURCE_DIR}")

file(GLOB PROTOTXT_FILES "${CMAKE_SOURCE_DIR}/predict_sin/*.prototxt")
message(STATUS ${PROTOTXT_FILES})
file(COPY ${PROTOTXT_FILES} DESTINATION ${CMAKE_BINARY_DIR}/predict_sin)

add_executable(predict_sin predict_sin.cpp DenseNet.cpp)
target_link_libraries(predict_sin caffe caffeproto protobuf caffe glog gflags ${Boost_LIBRARIES})