#!/usr/bin/env bash
set -ex


# This is the master script to execute the tests in this folder


# MNIST tanh tests: #filePath, netfile, epsilon, t_out, relaxFactor, numLayer
python -u "../vnn-comp/2020/NLN/NNV/tanh/executeTanh.py" '../vnn-comp/2020/NLN/NNV/tanh/' 'tansig_200_50_nnv.mat' 5 900 0 2
python -u "../vnn-comp/2020/NLN/NNV/tanh/executeTanh.py" '../vnn-comp/2020/NLN/NNV/tanh/' 'tansig_200_50_nnv.mat' 12 900 0 2
python -u "../vnn-comp/2020/NLN/NNV/tanh/executeTanh.py" '../vnn-comp/2020/NLN/NNV/tanh/' 'tansig_200_100_50_nnv.mat' 5 900 0 3
python -u "../vnn-comp/2020/NLN/NNV/tanh/executeTanh.py" '../vnn-comp/2020/NLN/NNV/tanh/' 'tansig_200_100_50_nnv.mat' 12 900 0 3

# MNIST sigmoid tests:
python -u "../vnn-comp/2020/NLN/NNV/sigmoid/executeSig.py" '../vnn-comp/2020/NLN/NNV/sigmoid/' 'logsig_200_50_nnv.mat' 5 900 0 2
python -u "../vnn-comp/2020/NLN/NNV/sigmoid/executeSig.py" '../vnn-comp/2020/NLN/NNV/sigmoid/' 'logsig_200_50_nnv.mat' 12 900 0 2
python -u "../vnn-comp/2020/NLN/NNV/sigmoid/executeSig.py" '../vnn-comp/2020/NLN/NNV/sigmoid/' 'logsig_200_100_50_nnv.mat' 5 900 0 3
python -u "../vnn-comp/2020/NLN/NNV/sigmoid/executeSig.py" '../vnn-comp/2020/NLN/NNV/sigmoid/' 'logsig_200_100_50_nnv.mat' 12 900 0 3