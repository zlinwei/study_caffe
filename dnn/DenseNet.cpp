//
// Created by root on 18-10-18.
//

#include "DenseNet.hpp"
#include <caffe/caffe.hpp>
#include <glog/logging.h>
#include <boost/random.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/normal_distribution.hpp>

void DenseNet::Initialize() {
    LOG(INFO) << "Initialize Dense Net";
    ::caffe::SolverParameter solver_param;
    ::caffe::ReadSolverParamsFromTextFileOrDie(_solver_prototxt_filename, &solver_param);

    _solver.reset(::caffe::SolverRegistry<float>::CreateSolver(solver_param));

    LOG(INFO) << _solver->type();
    _net = _solver->net();

    _data = _net->blob_by_name("data");
    assert(_data);

    _label = _net->blob_by_name("label");
    assert(_label);

    assert(CheckDataShape(_data, BATCH_SIZE, CHANNEL_SIZE, HEIGHT_SIZE, WIDTH_SIZE));

    _next_value = _net->blob_by_name("next_value");
    assert(_next_value);

    _input_layer = boost::dynamic_pointer_cast<::caffe::MemoryDataLayer<float>>(_net->layer_by_name("input_layer"));
    assert(_input_layer);

    _loss = _net->blob_by_name("loss");
    assert(_loss);

    LOG(INFO) << "loss shape: " << _loss->shape_string();

}

bool DenseNet::CheckDataShape(const DenseNet::BlobSptr &blob, int batch_size, int channel_size, int height, int width) {
    return blob->num() == batch_size && blob->channels() == channel_size && blob->height() == height &&
           blob->width() == width;
}

void DenseNet::TrainNet() {
    InputDataType input{};
    OutputDataType label{};
    std::fill(input.begin(), input.end(), 0.0f);
    std::fill(label.begin(), label.end(), 0.0f);


    boost::random::mt19937 rng(static_cast<const uint32_t &>(_current_iter + time(0)));
    boost::random::uniform_real_distribution<float> uniform(0, 10);

    boost::random::mt19937 rng1(static_cast<const uint32_t &>(_current_iter + time(0) + 1));
    boost::random::normal_distribution<float> error(0, 0.01);

    for (int j = 0; j < BATCH_SIZE; ++j) {
        input[j] = uniform(rng);
        label[j] = k * input[j] + b + error(rng1);
    }

    _input_layer->Reset(input.begin(), label.begin(), BATCH_SIZE);

    _current_iter++;

    _solver->Step(100);
}


float DenseNet::Loss() {
    return _loss->data_at(0, 0, 0, 0);
}

void DenseNet::TestNet() {
    InputDataType input{};
    OutputDataType output{};
    //y=k*x+b
    boost::random::mt19937 rng(static_cast<const uint32_t &>(time(0)));
    boost::random::uniform_real_distribution<float> uniform;
    for (int j = 0; j < BATCH_SIZE; ++j) {
        input[j] = uniform(rng);
        output[j] = k * input[j] + b;
    }

    _input_layer->Reset(input.begin(), output.begin(), BATCH_SIZE);

    float loss;
    _net->Forward(&loss);

    std::ofstream outfile("test_result.txt", std::ios::trunc);
    for (int k = 0; k < BATCH_SIZE; ++k) {
        outfile << _data->data_at(k, 0, 0, 0) << " " << output[k] << " " << _next_value->data_at(k, 0, 0, 0) << "\n";
    }
    outfile.close();
}

void DenseNet::Snapshot() {
    _solver->Snapshot();
}
