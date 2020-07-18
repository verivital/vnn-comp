"""
Small script for benchmarking.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os

import numpy as np

from evaluation.benchmark_scripts.benchmark import run_benchmark
from src.data_loader.input_data_loader import load_images_eran

if __name__ == "__main__":

    epsilons = [2/255]
    timeout = 300

    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((1, 3, 1, 1))
    std = np.array([0.2023, 0.1994, 0.2010]).reshape((1, 3, 1, 1))

    num_images = 100

    base_path = os.path.join(os.path.dirname(__file__), "../../../")
    img_dir = base_path + "2020/CNN/benchmark/cifar/eth/cifar10_test.csv"
    model_path = base_path + "2020/CNN/benchmark/cifar/eth/cifar10_2_255.onnx"
    result_path = base_path + "verinet/evaluation/benchmark_results/cifar10_2_255.txt"

    if not os.path.isdir(base_path + "verinet/evaluation/benchmark_results"):
        os.mkdir(base_path + "verinet/evaluation/benchmark_results")

    images, labels = load_images_eran(img_csv=img_dir)
    images = images.reshape(100, 32, 32, 3).transpose(0, 3, 1, 2)/255

    run_benchmark(images=images,
                  targets=labels,
                  epsilons=epsilons,
                  timeout=timeout,
                  mean=mean,
                  std=std,
                  model_path=model_path,
                  result_path=result_path,
                  max_procs=10)
