//
// Created by root on 18-10-18.
//

#include "DenseNet.hpp"
#include <caffe/caffe.hpp>
#include <glog/logging.h>

void DenseNet::Initialize() {
    LOG(INFO) << "Initialize Dense Net";
    ::caffe::SolverParameter solver_param;
    ::caffe::ReadSolverParamsFromTextFileOrDie(_solver_prototxt_filename, &solver_param);

    _solver.reset(::caffe::SolverRegistry<float>::CreateSolver(solver_param));
    _net = _solver->net();

    _data = _net->blob_by_name("data");
    assert(_data);

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

void DenseNet::PutData(InputDataType data, OutputDataType label) {
    _input_layer->Reset(data.begin(), label.begin(), BATCH_SIZE);
}

void DenseNet::UpdateNet() {
    InputDataType input{};
    OutputDataType label{};
    std::fill(input.begin(), input.end(), 0.0f);

    for (int i = 0; i < BATCH_SIZE; ++i) {
        std::copy(_train_data.begin() + i + _current_iter * 10,
                  _train_data.begin() + WIDTH_SIZE + i + _current_iter * 10,
                  input.begin() + i * WIDTH_SIZE);

        label[i] = _train_data[i + WIDTH_SIZE + _current_iter * 10];
    }

    if (_current_iter == 0)
        for (int i = 0; i < BATCH_SIZE; ++i) {
            LOG(INFO) << input[i * WIDTH_SIZE + WIDTH_SIZE] << " " << label[i];
        }

    _current_iter++;

    this->PutData(input, label);
    _net->ForwardBackward();


}

void DenseNet::InitTrainData() {
    for (int i = 0; i < _train_data.size(); ++i) {
        _train_data[i] = sin(float(i) / 10.0f);
    }
}

float DenseNet::Loss() {
    return _loss->data_at(0, 0, 0, 0);
}

void DenseNet::TestNet() {
    InputDataType input{};
    OutputDataType output{};
    std::fill(output.begin(), output.end(), 0.0f);
    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < WIDTH_SIZE; ++j) {
            input[i * WIDTH_SIZE + j] = sin(float(j + i) / 10.0f);
        }
    }


    PutData(input, output);
    _net->Forward(nullptr);

    LOG(INFO) << "result shape: " << _next_value->shape_string();
    std::ofstream outfile("values.txt", std::ios::trunc);
    for (int k = 0; k < BATCH_SIZE; ++k) {
        outfile << input[k] << " " << _data->data_at(k, 0, 0, 0) << " " << _next_value->data_at(k, 0, 0, 0) << "\n";
    }
    outfile.close();
}
