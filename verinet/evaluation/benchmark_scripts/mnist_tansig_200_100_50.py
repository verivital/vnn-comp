"""
Small script for benchmarking

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os

from evaluation.benchmark_scripts.benchmark import run_benchmark
from src.data_loader.input_data_loader import load_mnist_human_readable

if __name__ == "__main__":

    epsilons = [3, 5, 12]
    timeout = 900

    mean = 0
    std = 1

    num_images = 16

    base_path = os.path.join(os.path.dirname(__file__), "../../../")
    img_dir = base_path + "2020/PWL/benchmark/mnist/oval/mnist_images/"
    model_path = base_path + "verinet/data/tansig_200_100_50_onnx.onnx"
    result_path = base_path + "verinet/evaluation/benchmark_results/mnist_tansig_200_100_50.txt"

    if not os.path.isdir(base_path + "verinet/evaluation/benchmark_results"):
        os.mkdir(base_path + "verinet/evaluation/benchmark_results")

    images = load_mnist_human_readable(img_dir, list(range(1, num_images + 1))).reshape(num_images, -1)

    with open(img_dir+"labels", "r") as f:
        labels = [float(label) for label in f.readline().split(",")[:-1]]

    run_benchmark(images=images,
                  targets=labels,
                  epsilons=epsilons,
                  timeout=timeout,
                  mean=mean,
                  std=std,
                  model_path=model_path,
                  result_path=result_path,
                  max_val=255,
                  gradient_step=10)
