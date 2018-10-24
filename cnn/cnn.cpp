//
// Created by root on 18-10-22.
//
#include <iostream>
#include "DeepNet.hpp"

int main(){
    ::caffe::Caffe::set_mode(::caffe::Caffe::GPU);
    auto *deepNet = new DeepNet<float>();
    deepNet->Init();
    deepNet->InitLayer();

    return 0;
}