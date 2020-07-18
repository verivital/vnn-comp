"""
Small script for benchmarking

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import os

from evaluation.benchmark_scripts.benchmark import run_benchmark
from src.data_loader.input_data_loader import load_images_eran

if __name__ == "__main__":

    epsilons = [0.1]
    timeout = 300

    mean = 0.1307
    std = 0.3081

    num_images = 100

    base_path = os.path.join(os.path.dirname(__file__), "../../../")
    img_dir = base_path + "2020/CNN/benchmark/mnist/eth/mnist_test.csv"
    model_path = base_path + "2020/CNN/benchmark/mnist/eth/mnist_0.1.onnx"
    result_path = base_path + "verinet/evaluation/benchmark_results/mnist_0.1.txt"

    if not os.path.isdir(base_path + "verinet/evaluation/benchmark_results"):
        os.mkdir(base_path + "verinet/evaluation/benchmark_results")

    images, labels = load_images_eran(img_csv=img_dir, image_shape=(28, 28))
    images = images.reshape(100, 1, 28, 28)/255

    run_benchmark(images=images,
                  targets=labels,
                  epsilons=epsilons,
                  timeout=timeout,
                  mean=mean,
                  std=std,
                  model_path=model_path,
                  result_path=result_path)
