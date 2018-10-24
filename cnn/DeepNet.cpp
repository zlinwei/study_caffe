//
// Created by root on 18-10-23.
//

#include "DeepNet.hpp"

template<typename Dtype>
void DeepNet<Dtype>::Init() {
    ::caffe::SolverParameter solver_param;
    ::caffe::ReadProtoFromTextFileOrDie(_solver_prototxt, &solver_param);
    _solver.reset(::caffe::SolverRegistry<Dtype>::CreateSolver(solver_param));
    assert(_solver);
    _net = _solver->net();
}

template<typename Dtype>
void DeepNet<Dtype>::InitLayer() {
    for (auto i :  _net->layer_names()) {
        LOG(INFO) << i;
    }
    _input_data_layer = boost::dynamic_pointer_cast<::caffe::MemoryDataLayer<Dtype>>(
            _net->layer_by_name(_input_data_layer_name));
    assert(_input_data_layer);

    _input_label_layer = boost::dynamic_pointer_cast<::caffe::MemoryDataLayer<Dtype>>(_net->layer_by_name(_input_label_layer_name));
    assert(_input_data_layer);
}


INSTANTIATE_CLASS(DeepNet);