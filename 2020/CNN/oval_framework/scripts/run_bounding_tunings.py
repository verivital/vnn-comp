import os

"""
Run the experiments to tune competing optimizers for the planet problem w/ variable splitting
"""

def run_incomplete_verif_nets(gpu_id, cpus, iters, define_linear_approximation, images, adam_algorithms, prox_algorithms, data):

    define_linear_approximation_string = "--define_linear_approximation" if define_linear_approximation else ""

    for img in images:
        for algo, beta1, inlr, finlr in adam_algorithms:

            # have adam run for more as it's faster
            adam_iters = int(iters * 2.6) if algo == "dj-adam" else int(iters * 1.85)

            command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python tools/bounding_tools/bounding_runner.py " \
                f"--eps 0.04705882352 --algorithm {algo} --out_iters {adam_iters} --img_idx" \
                f" {img} --init_step {inlr} --fin_step {finlr} --beta1 {beta1} {define_linear_approximation_string} " \
                f" --data {data} --network_filename ./data/cifar_madry_8px.pth"
            print(command)
            os.system(command)

        for algo, momentum, ineta, fineta in prox_algorithms:
            command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python tools/bounding_tools/bounding_runner.py " \
                f"--eps 0.04705882352 --algorithm {algo} --out_iters {iters} " \
                f"--img_idx {img} --eta {ineta} --feta {fineta} --prox_momentum {momentum} " \
                f"{define_linear_approximation_string} --data {data} " \
                f"--network_filename ./data/cifar_madry_8px.pth" 
            print(command)
            os.system(command)


def run_complete_verif_nets(gpu_id, cpus, iters, define_linear_approximation, net, pdprops, prop_idxs, adam_algorithms, prox_algorithms, data):

    define_linear_approximation_string = "--define_linear_approximation" if define_linear_approximation else ""
    pdprops_str = f"--pdprops {pdprops}" if pdprops else ""

    for prop_idx in prop_idxs:
        for algo, beta1, inlr, finlr in adam_algorithms:
            # have adam run for more as it's faster
            adam_iters = int(iters * 2.6) if algo == "dj-adam" else int(iters * 1.6)

            command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python tools/bounding_tools/bounding_runner.py " \
                      f"--nn_name {net} --prop_idx {prop_idx} {pdprops_str} --algorithm {algo} " \
                      f"--out_iters {adam_iters} --data {data}" \
                      f" --init_step {inlr} --fin_step {finlr} --beta1 {beta1} {define_linear_approximation_string}"
            print(command)
            os.system(command)

        for algo, momentum, ineta, fineta in prox_algorithms:
            command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python tools/bounding_tools/bounding_runner.py " \
                f"--nn_name {net} --prop_idx {prop_idx} {pdprops_str} --algorithm {algo} --out_iters {iters} " \
                f"--eta {ineta} --feta {fineta} --prox_momentum {momentum} {define_linear_approximation_string} --data {data}"
            print(command)
            os.system(command)


if __name__ == "__main__":

    gpu_id = 0  # 0
    cpus = "0-5"  # "10-14"

    iters = 100

    define_linear_approximation = False

    # images = [0, 5, 10, 15, 20]
    adam_algorithms = [
        # algo, beta1, inlr, finlr
        ("planet-adam", 0.9, 1e-2, 1e-4),
        ("planet-adam", 0.9, 1e-3, 1e-6),
        ("planet-adam", 0.9, 1e-4, 1e-6),
        ("planet-adam", 0.9, 1e-3, 1e-4),
        ("planet-adam", 0.9, 1e-2, 1e-2),
        ("planet-adam", 0.9, 1e-3, 1e-3),
        ("planet-adam", 0.9, 1e-4, 1e-4),
        ("planet-adam", 0.9, 1e-1, 1e-4),
        ("planet-adam", 0.9, 1e-1, 1e-3),
    ]
    prox_algorithms = [  # momentum tuning
        # algo, momentum, ineta, fineta
        ("proxlp", 0.0, 1e2, 1e2),
        ("proxlp", 0.0, 1e1, 1e1),
        ("proxlp", 0.0, 1e3, 1e3),
        ("proxlp", 0.0, 1e0, 1e0),
        ("proxlp", 0.0, 1e4, 1e4),
        ("proxlp", 0.0, 5e1, 1e2),
        ("proxlp", 0.0, 5e2, 1e3),
        ("proxlp", 0.0, 5e0, 1e1),
        ("proxlp", 0.0, 5e3, 1e4),
    ]

    # Incomplete verification experiments (UAI net)
    # run_incomplete_verif_nets(gpu_id, cpus, iters, define_linear_approximation, images, adam_algorithms, prox_algorithms, data, net)

    # Complete verification experiments (BaB nets)
    # data = "cifar"
    # net = "cifar_base_kw"
    # pdprops = "jodie-base_hard.pkl"

    # run_incomplete_verif_nets function to be used for mnist-eth
    data = "mnist"
    net = "mnist_0.3"  # mnist_0.1 or mnist_0.3
    pdprops = None
    # images = list(range(100)) # all 100 images for workshop

    # cifar-eth
    data = "cifar10"
    net = "cifar10_8_255"  # cifar10_8_255 or cifar10_2_255
    pdprops = None
    if data == 'cifar10':
        with open(f"./data/correct-{net}.txt", "r") as file:
            line = file.readline()
        images = line.split(", ")[:-1]
        images = [int(img) for img in images]
        run_complete_verif_nets(gpu_id, cpus, iters, define_linear_approximation, net, pdprops, images, adam_algorithms, prox_algorithms, data)

    # read images from file
    try:
        with open(f"./data/undecided-{data}-eps{net[6:]}.txt", "r") as file:
            line = file.readline()
        images = line.split(", ")[:-1]
        images = [int(img) for img in images]
        if len(images) > 5:
            images = images[:5]

        run_complete_verif_nets(gpu_id, cpus, iters, define_linear_approximation, net, pdprops, images, adam_algorithms, prox_algorithms, data)
    except FileNotFoundError:
        print("No undecided images for the dataset.")
