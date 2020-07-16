#!/usr/bin/env python3

import argparse
import os
from enum import Enum

import onnx
import onnx.numpy_helper

import numpy as np
import scipy.io as sio

"""
Converts saved `.onnx` files of fully connected networks to `.mat` files in a
format we can process.
"""
def reorder_dims(xs):
    if len(xs.shape) == 4:
        xs = np.transpose(xs, [2, 3, 1, 0])
    if len(xs.shape) == 2:
        xs = np.transpose(xs)
    return xs


def convert_onnx_to_mat(input_path, output_path):
    print("Reading from {} and writing out to {}".format(input_path, output_path))
    model = onnx.load(input_path)
    d = {t.name: reorder_dims(onnx.numpy_helper.to_array(t)) for t in model.graph.initializer}

    print("Layers extracted: {}".format(list(d)))
    sio.savemat(output_path, d)


def get_path(relative_path):
    return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_path))

pat_fc_networks = {
    "../../../benchmark/mnist/oval/mnist-net_256x2.onnx": "../../networks/mnist-net_256x2.mat",
    "../../../benchmark/mnist/oval/mnist-net_256x4.onnx": "../../networks/mnist-net_256x4.mat",
    "../../../benchmark/mnist/oval/mnist-net_256x6.onnx": "../../networks/mnist-net_256x6.mat",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract weights from .onnx file for a fully connected network."
    )

    for relative_input_path, relative_output_path in pat_fc_networks.items():
        input_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_input_path))
        output_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_output_path))
        convert_onnx_to_mat(input_path, output_path)
