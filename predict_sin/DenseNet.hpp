//
// Created by root on 18-10-18.
//

#ifndef STUDY_CAFFE_DENSENET_HPP
#define STUDY_CAFFE_DENSENET_HPP

#include <string>
#include <array>

#include <boost/shared_ptr.hpp>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

constexpr auto BATCH_SIZE = 128;
constexpr auto CHANNEL_SIZE = 1;
constexpr auto HEIGHT_SIZE = 1;
constexpr auto WIDTH_SIZE = 100;

constexpr auto OUTPUT_SIZE = 1;

constexpr auto INPUT_SIZE = CHANNEL_SIZE * HEIGHT_SIZE * WIDTH_SIZE;

class DenseNet {
public:
    void Initialize();

    using SolverSptr = boost::shared_ptr<::caffe::Solver<float>>;

    using NetSptr = boost::shared_ptr<::caffe::Net<float>>;

    using BlobSptr = boost::shared_ptr<::caffe::Blob<float>>;

    using MemoryDataLayerSptr = boost::shared_ptr<::caffe::MemoryDataLayer<float>>;

    using InputDataType = std::array<float, INPUT_SIZE * BATCH_SIZE>;

    using OutputDataType = std::array<float, OUTPUT_SIZE * BATCH_SIZE>;

    void UpdateNet();

    void PutData(InputDataType, OutputDataType);

    void InitTrainData();

    float Loss();

    void TestNet();

private:
    bool CheckDataShape(const BlobSptr &, int, int, int, int);

private:
    std::string _solver_prototxt_filename = "solver.prototxt";

    SolverSptr _solver;

    NetSptr _net;

    BlobSptr _data;

    BlobSptr _next_value;

    BlobSptr _loss;

    MemoryDataLayerSptr _input_layer;

    long long int _current_iter = 0;

    std::array<float, WIDTH_SIZE * 100> _train_data;
};


#endif //STUDY_CAFFE_DENSENET_HPP
