//
// Created by linwei on 18-10-23.
//

#ifndef DEEPNET_HPP
#define DEEPNET_HPP

#include <iostream>
#include <vector>

#include <boost/shared_ptr.hpp>

#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

template<typename Dtype>
class DeepNet {
public:
//    using SolverSptr = boost::shared_ptr<::caffe::Solver<T>>;

//    using MemoryDataLayerSptr = boost::shared_ptr<::caffe::MemoryDataLayer<T>>;

    void Init();

private:
//    std::string _solver_prototxt = "solver.prototxt";
//
//    SolverSptr _solver;
//
//    MemoryDataLayerSptr _input_data_layer;
//
//    MemoryDataLayerSptr _input_label_layer;


};


#endif //DEEPNET_HPP
