"""
Small script to fix the nln models.

The onnx models have wrongly stored fc-layers as conv, this scripts converts the
conv layers to FC.
"""

import os

import torch
import torch.onnx as tonnx
import torch.nn as nn

from src.neural_networks.verinet_nn import VeriNetNN
from src.data_loader.onnx_parser import ONNXParser

if __name__ == '__main__':

    in_base_path = os.path.join(os.path.dirname(__file__), "../../../2020/NLN/benchmark/mnist/")
    out_base_path = os.path.join(os.path.dirname(__file__), "../../data/")

    org_files = [ "sigmoid/logsig_200_50_onnx.onnx",
                  "sigmoid/logsig_200_100_50_onnx.onnx",
                  "tanh/tansig_200_50_onnx.onnx",
                  "tanh/tansig_200_100_50_onnx.onnx"]

    for file in org_files:

        onnx_parser = ONNXParser(os.path.join(in_base_path, file))
        model = onnx_parser.to_pytorch()

        layers_org = model.layers
        layers_fc = []
        print(layers_org)

        for layer in layers_org:

            if isinstance(layer, nn.Conv2d):
                weights = layer.weight.data.squeeze()
                bias = layer.bias.data.squeeze()

                layer_fc = nn.Linear(in_features=weights.shape[1], out_features=weights.shape[0])
                layer_fc.weight.data = weights
                layer_fc.bias.data = bias

                layers_fc.append(layer_fc)

            else:
                layers_fc.append(layer)

        model_fc = VeriNetNN(layers_fc)

        dummy_in = torch.randn(1, 784, 1)
        tonnx.export(model_fc, dummy_in, os.path.join(out_base_path, file.split("/")[-1]), verbose=True, opset_version=9)




