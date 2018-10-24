//
// Created by root on 18-10-22.
//
#include <iostream>
#include "DeepNet.hpp"

int main(){
#ifdef CPU_ONLY
    ::caffe::Caffe::set_mode(::caffe::Caffe::CPU);
#else
    ::caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif

    LOG(INFO) << "start..";
    auto *deepNet = new DeepNet<float>();
    deepNet->Init();
    deepNet->InitLayer();

    return 0;
}