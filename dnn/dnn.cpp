//
// Created by root on 18-10-18.
//
#include <fstream>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <caffe/caffe.hpp>

#include "DenseNet.hpp"

int main(int argc, char *argv[]) {
#ifdef CPU_ONLY
    ::caffe::Caffe::set_mode(::caffe::Caffe::CPU);
#else
    ::caffe::Caffe::set_mode(caffe::Caffe::GPU);
#endif
    LOG(INFO) << "start..";

    boost::shared_ptr<DenseNet> net = boost::make_shared<DenseNet>();

    net->Initialize();

    for (int j = 0; j < 10; ++j) {
        net->TrainNet();
        LOG(INFO) << "loss: " << net->Loss();
    }


    net->TestNet();

    net->Snapshot();

    return 0;
}
