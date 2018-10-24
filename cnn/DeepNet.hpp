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
    using SolverSptr = boost::shared_ptr<::caffe::Solver<Dtype>>;

    using NetSptr = boost::shared_ptr<::caffe::Net<Dtype>>;

    using MemoryDataLayerSptr = boost::shared_ptr<::caffe::MemoryDataLayer<Dtype>>;

    virtual void Init();

    virtual void InitLayer();

private:
    std::string _solver_prototxt = "solver.prototxt";

    std::string _input_data_layer_name = "input_data";

    std::string _input_label_layer_name = "input_label";

    SolverSptr _solver;

    NetSptr _net;

    MemoryDataLayerSptr _input_data_layer;

    MemoryDataLayerSptr _input_label_layer;


};



#endif //DEEPNET_HPP
