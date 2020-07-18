import os, sys

"""
Generating training data sets
"""


def run_traingen(gpu_id, cpus, pdprops, nn, method, data, gt_throughout):

    if gt_throughout:
        pdprops_list = [
            "trainfull_sub1.pkl",
            "trainfull_sub2.pkl",
            "val_full_props.pkl",
            "sanitycheck_fake2_timeout_props.pkl",
            "trainfake_sub1.pkl",
            "trainfake_sub2.pkl",
        ]
    else:
        pdprops_list = [
            "trainfake_sub1.pkl",
            "trainfake_sub2.pkl",
            "val_fake_props.pkl",
        ]

    nn_names = [
        "cifar_base_kw",
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
            alg_specs += f" --n_cuts 100"
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
    if nn in ["cifar_base_kw", "mnist_0.1", "mnist_0.3"]:
        max_solver_batch = 25000
    else:
        max_solver_batch = 18000

    command = f"CUDA_VISIBLE_DEVICES={gpu_id} taskset -c {cpus} python tools/bab_tools/bab_runner.py " \
              f" {pdprops_str} --nn_name {nn} --record --method {method} {alg_specs} " \
              f"--batch_size {batch_size} --max_solver_batch {max_solver_batch} {parent_init} --data {data} --train_generation"
    if gt_throughout:
        command += ' --train_gt_throughout'
    print(command)
    os.system(command)


if __name__ == "__main__":

    #pdprops = "trainfull_sub1.pkl"
    #pdprops = "trainfull_sub2.pkl"
    #pdprops = "val_full_props.pkl"
    #pdprops = "trainfake_sub1.pkl"
    pdprops = "trainfake_sub2.pkl"
    #pdprops = "val_fake_props.pkl"
    #pdprops = "sanitycheck_fake2_timeout_props.pkl"
    data = "cifar"
    nn = "cifar_base_kw"
    method = 'prox'
    gt_throughout = True

    #gpu_id = 0
    #cpus = "0-5"  # 4 cpus needed
    #gpu_id = 1
    #cpus = "6-10"  # 4 cpus needed
    gpu_id = 2
    cpus = "11-15"  # 4 cpus needed

    run_traingen(gpu_id, cpus, pdprops, nn, method, data, gt_throughout)
