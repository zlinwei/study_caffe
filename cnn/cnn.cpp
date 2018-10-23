//
// Created by root on 18-10-22.
//
#include <iostream>
#include "DeepNet.hpp"

int main(){
    auto *deepNet = new DeepNet<float>();
    deepNet->Init();

    return 0;
}