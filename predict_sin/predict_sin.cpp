//
// Created by root on 18-10-18.
//
#include <fstream>

#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <caffe/caffe.hpp>

#include "DenseNet.hpp"

int main(int argc, char *argv[]) {
    ::caffe::Caffe::set_mode(caffe::Caffe::CPU);

    LOG(INFO) << "start..";

    boost::shared_ptr<DenseNet> net = boost::make_shared<DenseNet>();

    net->Initialize();

    net->InitTrainData();

    for (int i = 0; i < 500; ++i) {
        net->UpdateNet();
        if (i % 10 == 0) {
            LOG(INFO) << net->Loss() << " " << i;
        }
    }

    net->TestNet();

    return 0;
}
