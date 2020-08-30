#!/usr/bin/env python3

import argparse
import os

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
    d = {
        t.name: reorder_dims(onnx.numpy_helper.to_array(t))
        for t in model.graph.initializer
    }

    print("Layers extracted: {}".format(list(d)))
    sio.savemat(output_path, d)


def get_path(relative_path):
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), relative_path)
    )


networks = {
    "../../../benchmark/cifar/eth/cifar10_2_255.onnx": "../../networks/cifar10_2_255.mat",
    "../../../benchmark/cifar/eth/cifar10_8_255.onnx": "../../networks/cifar10_8_255.mat",
    "../../../benchmark/mnist/eth/mnist_0.1.onnx": "../../networks/mnist_0.1.mat",
    "../../../benchmark/mnist/eth/mnist_0.3.onnx": "../../networks/mnist_0.3.mat",
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract weights from .onnx file for a fully connected network."
    )

    for relative_input_path, relative_output_path in networks.items():
        input_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), relative_input_path
            )
        )
        output_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), relative_output_path
            )
        )
        convert_onnx_to_mat(input_path, output_path)
