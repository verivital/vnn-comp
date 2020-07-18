import os, sys

"""
Run the bab_runner experiments
"""


def run_bab_exp(gpu_id, cpus, timeout, pdprops, nn, method, data, branching=None):

    pdprops_list = [
        "jodie-base_easy.pkl",
        "jodie-base_med.pkl",
        "jodie-base_hard.pkl",
        "jodie-wide.pkl",
        "jodie-deep.pkl"
    ]
    nn_names = [
        "cifar_base_kw",
        "cifar_wide_kw",
        "cifar_deep_kw",
        "mnist_0.1",
        "mnist_0.3",
        ]
    methods = [
        "prox",
        "adam",
        "dj-adam",
        "gurobi",
        "gurobi-anderson",
    ]
    if pdprops:
        assert pdprops in pdprops_list
    assert nn in nn_names
    assert method in methods
    pdprops_str = f"--pdprops {pdprops}" if pdprops else ""

    if method in ["gurobi", "gurobi-anderson"]:
        batch_size = 150
        parent_init = ""
        alg_specs = "--gurobi_p 6"  # Gurobi runs on 6 cores
        if method == "gurobi-anderson":
            alg_specs += f" --n_cuts 1"
    else:
        parent_init = "--parent_init"
        if nn in ["cifar_base_kw", "mnist_0.1", "mnist_0.3"]:
            # TODO: potentially, this could be larger for MNIST
            batch_size = 150
        else:
            batch_size = 100
        if method == "prox":
            alg_specs = "--tot_iter 100"
            if "mnist" in nn:
                alg_specs += " --eta 1e1 --feta 1e1"
            else:
                alg_specs += " --eta 1e2 --feta 1e2"
        elif method == "adam":
            adam_iters = 175 if "mnist" in nn else 160
            alg_specs = f"--tot_iter {adam_iters}"
            if "mnist" in nn:
                alg_specs += " --init_step 1e-1 --fin_step 1e-3"
            else:
                alg_specs += " --init_step 1e-3 --fin_step 1e-4"
        elif method == "dj-adam":
            alg_specs = "--tot_iter 260"
            if "mnist" in nn:
                alg_specs += " --init_step 1e-1 --fin_step 1e-3"
            else:
                alg_specs += " --init_step 1e-3 --fin_step 1e-4"

    # TODO: potentially, this could be larger for MNIST
    memory_limit = 500 if nn in ["cifar_base_kw", "mnist_0.1", "mnist_0.3"] else 300

    command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python tools/bab_tools/bab_runner.py " \
              f"--timeout {timeout} {pdprops_str} --nn_name {nn} --record --method {method} {alg_specs} " \
              f"--batch_size {batch_size} --max_mem_consumption {memory_limit} {parent_init} --data {data}"
    if branching == 'gnn':
        command += ' --branching_choice gnn'
    else:
        command += ' --branching_choice heuristic'
    print(command)
    os.system(command)


if __name__ == "__main__":

    pdprops = "jodie-base_easy.pkl"
    data = "cifar"
    nn = "cifar_base_kw"
    timeout = 3600
    method = "prox"
    gpu_id = 1
    #branching = 'gnn'
    branching = 'heuristic'
    cpus = "12-17"  # 4 cpus needed
    run_bab_exp(gpu_id, cpus, timeout, pdprops, nn, method, data, branching)

    #data = "mnist"
    #nns = ["mnist_0.1", "mnist_0.3"]
    #pdprops = None
    #timeout = 600  # 10-min timeout (can be clipped afterwards)

    # run experiments on helios
    #if int(sys.argv[1]) == 0:
    #    methods = ["prox", "gurobi-anderson"]
    #    gpu_id = 0
    #    cpus = "6-11"  # 4 cpus needed
    #elif int(sys.argv[1]) == 1:
    #    methods = ["adam", "gurobi"]
    #    gpu_id = 1
    #    cpus = "0-5"  # 4 cpus needed
    #else:
    #    raise IOError("Wrong input parameter: must be in [0, 1].")

    #for method in methods:
    #    for nn in nns:
    #        run_bab_exp(gpu_id, cpus, timeout, pdprops, nn, method, data)
