#!/usr/bin/env bash
set -ex

# This is the master script to execute the tests in this folder


# MNIST ReLU FCN tests: #filePath, netfile, epsilon, t_out, relaxFactor, numLayer
python -u "../vnn-comp/2020/PWL/NNV/MnistNets/executeImperial.py" '../vnn-comp/2020/PWL/NNV/MnistNets/' 'net256x2.mat' 0.02 900 0 2
python -u "../vnn-comp/2020/PWL/NNV/MnistNets/executeImperial.py" '../vnn-comp/2020/PWL/NNV/MnistNets/' 'net256x2.mat' 0.05 900 0 2
python -u "../vnn-comp/2020/PWL/NNV/MnistNets/executeImperial.py" '../vnn-comp/2020/PWL/NNV/MnistNets/' 'net256x4.mat' 0.02 900 0 4
python -u "../vnn-comp/2020/PWL/NNV/MnistNets/executeImperial.py" '../vnn-comp/2020/PWL/NNV/MnistNets/' 'net256x4.mat' 0.05 900 0 4
python -u "../vnn-comp/2020/PWL/NNV/MnistNets/executeImperial.py" '../vnn-comp/2020/PWL/NNV/MnistNets/' 'net256x6.mat' 0.02 900 0 6
python -u "../vnn-comp/2020/PWL/NNV/MnistNets/executeImperial.py" '../vnn-comp/2020/PWL/NNV/MnistNets/' 'net256x6.mat' 0.05 900 0 6




# ACASXu tests: #filePath, property_id, n1(starting index for types of networks),N1(types of networks), n2, N2, numCores, t_out

## ACAS_all:
## property==1
python -u "../vnn-comp/2020/PWL/NNV/Acas_Xu/executeAcasXu.py" '../vnn-comp/2020/PWL/NNV/Acas_Xu/' 1 1 5 1 9 4 300
## property==2
python -u "../vnn-comp/2020/PWL/NNV/Acas_Xu/executeAcasXu.py" '../vnn-comp/2020/PWL/NNV/Acas_Xu/' 2 1 5 1 9 4 300
## property==3
python -u "../vnn-comp/2020/PWL/NNV/Acas_Xu/executeAcasXu.py" '../vnn-comp/2020/PWL/NNV/Acas_Xu/' 3 1 5 1 9 4 300
## property==4
python -u "../vnn-comp/2020/PWL/NNV/Acas_Xu/executeAcasXu.py" '../vnn-comp/2020/PWL/NNV/Acas_Xu/' 4 1 5 1 9 4 300

## ACAS_hard:
python -u "../vnn-comp/2020/PWL/NNV/Acas_Xu/executeAcasXu.py" '../vnn-comp/2020/PWL/NNV/Acas_Xu/' 1 4 4 6 6 4 21600 1
python -u "../vnn-comp/2020/PWL/NNV/Acas_Xu/executeAcasXu.py" '../vnn-comp/2020/PWL/NNV/Acas_Xu/' 1 4 4 8 8 4 21600 1
python -u "../vnn-comp/2020/PWL/NNV/Acas_Xu/executeAcasXu.py" '../vnn-comp/2020/PWL/NNV/Acas_Xu/' 2 3 3 3 3 4 21600 1
python -u "../vnn-comp/2020/PWL/NNV/Acas_Xu/executeAcasXu.py" '../vnn-comp/2020/PWL/NNV/Acas_Xu/' 2 4 4 2 2 4 21600 1
python -u "../vnn-comp/2020/PWL/NNV/Acas_Xu/executeAcasXu.py" '../vnn-comp/2020/PWL/NNV/Acas_Xu/' 2 4 4 9 9 4 21600 1
python -u "../vnn-comp/2020/PWL/NNV/Acas_Xu/executeAcasXu.py" '../vnn-comp/2020/PWL/NNV/Acas_Xu/' 2 5 5 3 3 4 21600 1
python -u "../vnn-comp/2020/PWL/NNV/Acas_Xu/executeAcasXu.py" '../vnn-comp/2020/PWL/NNV/Acas_Xu/' 3 3 3 6 6 4 21600 1
python -u "../vnn-comp/2020/PWL/NNV/Acas_Xu/executeAcasXu.py" '../vnn-comp/2020/PWL/NNV/Acas_Xu/' 3 5 5 1 1 4 21600 1
python -u "../vnn-comp/2020/PWL/NNV/Acas_Xu/executeAcasXu.py" '../vnn-comp/2020/PWL/NNV/Acas_Xu/' 7 1 1 9 9 4 21600 1
python -u "../vnn-comp/2020/PWL/NNV/Acas_Xu/executeAcasXu.py" '../vnn-comp/2020/PWL/NNV/Acas_Xu/' 9 3 3 3 3 4 21600 1