#!/usr/bin/env bash
set -ex

# This is the master script to execute the tests in this folder


# MNIST tests: #filePath netfile epsilon timeOut relaxFactor
python -u "../vnn-comp/2020/CNN/NNV/MNIST/executeMnist.py" '../vnn-comp/2020/CNN/NNV/MNIST/' 'mnist01.mat' 0.1 60 0
python -u "../vnn-comp/2020/CNN/NNV/MNIST/executeMnist.py" '../vnn-comp/2020/CNN/NNV/MNIST/' 'mnist01.mat' 0.3 300 0 

# CIFAR10 tests:
python -u "../vnn-comp/2020/CNN/NNV/CIFAR10/executeCifar.py" '../vnn-comp/2020/CNN/NNV/CIFAR10/' 'cifar_2_255.mat' 2 300 0
python -u "../vnn-comp/2020/CNN/NNV/CIFAR10/executeCifar.py" '../vnn-comp/2020/CNN/NNV/CIFAR10/' 'cifar_8_255.mat' 8 300 0 