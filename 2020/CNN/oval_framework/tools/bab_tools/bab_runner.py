import argparse
import torch
from plnn.branch_and_bound.relu_train_generation import relu_traingen
from plnn.branch_and_bound.relu_branch_and_bound import relu_bab
from tools.bab_tools.model_utils import load_cifar_1to1_exp, load_1to1_eth
from plnn.proxlp_solver.solver import SaddleLP
from plnn.proxlp_solver.dj_relaxation import DJRelaxationLP
from plnn.network_linear_approximation import LinearizedNetwork
from plnn.anderson_linear_approximation import AndersonLinearizedNetwork
import time
import pandas as pd
import os, copy
import math
import torch.multiprocessing as mp
import itertools
import csv
'''
Code from GNN_branching (author: Jodie)
This script supports following verifications methods.

Branch and Bound with a heuristic splitting strategy, developed based on Kolter and Wong's paper (--bab_kw)
'''



# Pre-fixed parameters
pref_branching_thd = 0.2
pref_online_thd = 2
pref_kwbd_thd = 20

models = {}
models[
    'cifar_base_hinge10_norm_wd1e-4'] = './gnn_branching/gnn_models/cifar/best_snapshot_None_0_val_acc_0.746_loss_val_0.1897_epoch_58.pt'


def bab(gt_prop, verif_layers, domain, return_dict, timeout, batch_size, method, tot_iter,  parent_init,
        args, gurobi_dict=None):
    epsilon = 1e-4
    decision_bound = 0
    gpu = True

    if gpu:
        cuda_verif_layers = [copy.deepcopy(lay).cuda() for lay in verif_layers]
        domain = domain.cuda()
    else:
        cuda_verif_layers = [copy.deepcopy(lay) for lay in verif_layers]

    # use best of naive interval propagation and KW as intermediate bounds
    intermediate_net = SaddleLP(cuda_verif_layers, store_bounds_primal=True, max_batch=args.max_solver_batch)
    intermediate_net.set_solution_optimizer('best_naive_kw', None)

    if method == "prox":
        bounds_net = SaddleLP(cuda_verif_layers, store_bounds_primal=True, max_batch=args.max_solver_batch)
        bounds_net.set_decomposition('pairs', 'KW')
        optprox_params = {
            'nb_total_steps': int(tot_iter),
            'max_nb_inner_steps': 2,  # this is 2/5 as simpleprox
            'initial_eta': args.eta,
            'final_eta': args.feta,
            'log_values': False,
            'outer_cutoff': args.cutoff,
            'maintain_primal': True
        }
        bounds_net.set_solution_optimizer('optimized_prox', optprox_params)
        print(f"Running prox with {tot_iter} steps")
    elif method == "adam":
        bounds_net = SaddleLP(cuda_verif_layers, store_bounds_primal=True, max_batch=args.max_solver_batch)
        bounds_net.set_decomposition('pairs', 'KW')
        adam_params = {
            'nb_steps': int(tot_iter),
            'initial_step_size': args.init_step,
            'final_step_size': args.fin_step,
            'betas': (0.9, 0.999),
            'outer_cutoff': args.cutoff,
            'log_values': False
        }
        bounds_net.set_solution_optimizer('adam', adam_params)
        print(f"Running adam with {tot_iter} steps")
    elif method == "dj-adam":
        adam_params = {
            'nb_steps': int(tot_iter),
            'initial_step_size': args.init_step,
            'final_step_size': args.fin_step,
            'betas': (0.9, 0.999),
            'outer_cutoff': args.cutoff,
            'log_values': False
        }
        bounds_net = DJRelaxationLP(cuda_verif_layers, params=adam_params, store_bounds_primal=True,
                                    max_batch=args.max_solver_batch)
        print(f"Running DJ-adam with {tot_iter} steps")
    elif method == "gurobi":
        bounds_net = LinearizedNetwork(verif_layers)
    elif method == "gurobi-anderson":
        bounds_net = AndersonLinearizedNetwork(
            verif_layers, mode="lp-cut", n_cuts=args.n_cuts, cuts_per_neuron=True, decision_boundary=decision_bound)

    # branching
    if args.branching_choice == 'heuristic':
        branching_net_name = None
    else:
        if method == 'prox':
            branching_net_name = models['cifar_base_hinge10_norm_wd1e-4']
        else: 
            raise NotImplementedError

    # try:
    with torch.no_grad():
        if args.train_generation:
            min_lb, min_ub, ub_point, nb_states = relu_traingen(gt_prop, intermediate_net, bounds_net, domain, decision_bound,
                                                           eps=epsilon, 
                                                           parent_init_flag=parent_init, gurobi_specs=gurobi_dict, gt_throughout = args.train_gt_throughout)
            fail_safe_ratio = None

        else:
            min_lb, min_ub, ub_point, nb_states, fail_safe_ratio = relu_bab(intermediate_net, bounds_net, branching_net_name, domain, decision_bound,
                                                           eps=epsilon, timeout=timeout,
                                                           batch_size=batch_size,
                                                           parent_init_flag=parent_init, gurobi_specs=gurobi_dict)

    if not (min_lb or min_ub or ub_point):
        return_dict["min_lb"] = None;
        return_dict["min_ub"] = None;
        return_dict["ub_point"] = None;
        return_dict["nb_states"] = nb_states
        return_dict["bab_out"] = "timeout"
        return_dict["fs_ratio"] = fail_safe_ratio
    else:
        return_dict["min_lb"] = min_lb.cpu()
        return_dict["min_ub"] = min_ub.cpu()
        return_dict["ub_point"] = ub_point.cpu()
        return_dict["nb_states"] = nb_states
        return_dict["fs_ratio"] = fail_safe_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true', help='file to save results')
    parser.add_argument('--train_generation', action='store_true', help='mode of generating training datasets')
    parser.add_argument('--train_gt_throughout', action='store_true', help='generate groud truths for each branch throughout a property verification')
    parser.add_argument('--record_name', type=str, help='file to save results')
    parser.add_argument('--pdprops', type=str, help='pandas table with all props we are interested in')
    parser.add_argument('--timeout', type=int)
    parser.add_argument('--cpus_total', type=int, help='total number of cpus used')
    parser.add_argument('--cpu_id', type=int, help='the index of the cpu from 0 to cpus_total')
    parser.add_argument('--nn_name', type=str, help='network architecture name')
    parser.add_argument('--data', type=str, default='cifar')
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--batch_size', type=int, help='batch size / 2 for how many domain computations in parallel')
    parser.add_argument('--gurobi_p', type=int, help='number of threads to use in parallelizing gurobi over domains', default=1)
    parser.add_argument('--method', type=str, choices=["prox", "adam", "gurobi", "gurobi-anderson", "dj-adam"], help='method to employ for bounds')
    parser.add_argument('--branching_choice', type=str, choices=['heuristic', 'gnn'], help='type of branching choice used')
    parser.add_argument('--tot_iter', type=float, help='how many total iters to use for the method', default=100)
    parser.add_argument('--max_solver_batch', type=float, default=10000, help='max batch size for bounding computations')
    parser.add_argument('--parent_init', action='store_true', help='whether to initialize the code from the parent')
    parser.add_argument('--n_cuts', type=int, help='number of anderson cuts to employ (per neuron)')
    parser.add_argument('--eta', type=float)
    parser.add_argument('--feta', type=float)
    parser.add_argument('--cutoff', type=float, default=0)
    parser.add_argument('--init_step', type=float)
    parser.add_argument('--fin_step', type=float)
    args = parser.parse_args()

    # initialize a file to record all results, record should be a pandas dataframe
    if args.data == 'cifar' or args.data=='cifar10':
        if args.train_generation:
            path = './cifar_train_pdtables/'
            result_path = './cifar_train_results/'
            dom_path = f'./cifar_kw_prox_m2_train_data/{args.pdprops[:-4]}/'
            if not os.path.exists(dom_path):
                os.makedirs(dom_path)
        else: 
            path = './batch_verification_results/'
            result_path = './cifar_results/'

        if not os.path.exists(result_path):
            os.makedirs(result_path)

    elif args.data == 'mnist':
        if args.train_generation:
            raise NotImplementedError
        else:
            path = './mnist_batch_verification_results/'
            result_path = './mnist_results/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
    else:
        raise NotImplementedError

    # load all properties
    if args.data=='mnist' or args.data=='cifar10':
        csvfile = open('././data/%s_test.csv'%(args.data), 'r')
        tests = list(csv.reader(csvfile, delimiter=','))
        batch_ids = range(100)
        enum_batch_ids = [(bid, None) for bid in batch_ids]
    elif args.data=='cifar':
        gt_results = pd.read_pickle(path + args.pdprops)
        bnb_ids = gt_results.index
        batch_ids = bnb_ids
        enum_batch_ids = enumerate(batch_ids)

    if args.record:
        if args.record_name is not None:
            record_name = args.record_name
        else:
            method_name = ''
            columns = ["Idx", "Eps", "prop"]

            parent_init = "-pinit" if args.parent_init else ""
            algo_string = ""
            if args.method == "prox":
                algo_string += f"-eta{args.eta}-feta{args.feta}"
            elif args.method in ["adam", "dj-adam"]:
                algo_string += f"-ilr{args.init_step},flr{args.fin_step}"
            if args.cutoff and args.cutoff > 0:
                algo_string += f"-cutoff{args.cutoff}"

            if args.method not in ["gurobi", "gurobi-anderson"]:
                algorithm_name = f"{args.method}_{int(args.tot_iter)}"
            elif args.method == "gurobi-anderson":
                algorithm_name = f'{args.method}_{args.n_cuts}'
            else:
                algorithm_name = f'{args.method}'
            
            # branching choices
            if args.branching_choice=='heuristic':
                method_name += f'KW_{algorithm_name}{parent_init}{algo_string}'
                columns += [f'BSAT_KW_{algorithm_name}', f'BBran_KW_{algorithm_name}', f'BTime_KW_{algorithm_name}']
            else:
                method_name += f'GNN_{algorithm_name}{parent_init}{algo_string}'
                columns += [f'BSAT_GNN_{algorithm_name}', f'BBran_GNN_{algorithm_name}', f'BTime_GNN_{algorithm_name}', f'FS_Ratio_{algorithm_name}']


            if args.data=='mnist' or args.data=='cifar10':
                base_name = f'{args.nn_name}'
            else:
                base_name = f'{args.pdprops[:-4]}'
            if args.train_generation:
                base_name = 'traindata_generation_'+base_name
            record_name = result_path + f'{base_name}_{method_name}.pkl'

        if os.path.isfile(record_name):
            graph_df = pd.read_pickle(record_name)
        else:
            indices = list(range(len(batch_ids)))

            graph_df = pd.DataFrame(index=indices, columns=columns)
            graph_df.to_pickle(record_name)

            # skip = False

    if args.method in ["gurobi", "gurobi-anderson"]:
        if args.gurobi_p > 1:
            mp.set_start_method('spawn')  # for some reason, everything hangs w/o this
    gurobi_dict = {"gurobi": args.method in ["gurobi", "gurobi-anderson"], "p": args.gurobi_p}

    for new_idx, idx in enum_batch_ids:
        # record_info
        if args.record:
            graph_df = pd.read_pickle(record_name)
            if pd.isna(graph_df.loc[new_idx]['Eps']) == False:
                # print(f'the {new_idx}th element is done')
                # skip = True
                continue
        # if skip == True:
        #    print(f'skip the {new_idx}th element')
        #    skip = False
        #    continue

        

        if args.data == 'cifar':

            imag_idx = gt_results.loc[idx]["Idx"]
            prop_idx = gt_results.loc[idx]['prop']
            eps_temp = gt_results.loc[idx]["Eps"]

            # skip the nan prop_idx or eps_temp (happens in wide.pkl, jodie's mistake, I guess)
            if (math.isnan(imag_idx) or math.isnan(prop_idx) or math.isnan(eps_temp)):
                continue

            x, verif_layers, test = load_cifar_1to1_exp(args.nn_name, int(imag_idx), int(prop_idx))
            # since we normalise cifar data set, it is unbounded now
            bounded = False
            assert test == prop_idx
            domain = torch.stack([x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp], dim=-1)
            linear = False
        elif args.data == 'mnist' or args.data=='cifar10':

            imag_idx = new_idx
            eps_temp = float(args.nn_name[6:]) if args.data == 'mnist' else float(args.nn_name.split('_')[1])/float(args.nn_name.split('_')[2])

            x, verif_layers, test = load_1to1_eth(args.data, args.nn_name, idx=imag_idx, test=tests, eps_temp=eps_temp,
                                                  max_solver_batch=args.max_solver_batch)
            if x is None:
                # handle misclassified images
                continue
            # since we normalise cifar data set, it is unbounded now
            bounded = False
            prop_idx = test
            domain = torch.stack([x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp], dim=-1)
            linear = False

        else:
            raise NotImplementedError

        ### BaB
        gt_prop = f'idx_{imag_idx}_prop_{prop_idx}_eps_{eps_temp}'
        print(gt_prop)
        if args.train_generation:
            gt_prop = f'{args.pdprops[:-4]}/'+gt_prop
            print('Training Data Generation Mode')
            print(f'data is stored with the prefix {gt_prop}')
        bab_start = time.time()
        # manager = multiprocessing.Manager()
        # return_dict = manager.dict()
        return_dict = dict()
        bab(gt_prop, verif_layers, domain, return_dict, args.timeout, args.batch_size, args.method, args.tot_iter, args.parent_init, args, gurobi_dict=gurobi_dict)

        bab_min_lb = return_dict["min_lb"]
        bab_min_ub = return_dict["min_ub"]
        bab_ub_point = return_dict["ub_point"]
        bab_nb_states = return_dict["nb_states"]
        bab_fs = return_dict["fs_ratio"]
        if bab_min_lb is None:
            if "bab_out" in return_dict:
                bab_out = return_dict["bab_out"]
            else:
                bab_out = 'grbError'
        else:
            if bab_min_lb >= 0:
                print("UNSAT")
                bab_out = "False"
            elif bab_min_ub < 0:
                # Verify that it is a valid solution
                print("SAT")
                bab_out = "True"
            else:
                if args.train_generation:
                    #if bab_nb_states>1500:
                    print('Traindata generation mode: Early Termination')
                    bab_out = 'ET'
                else:
                    print("Unknown")
                    #import pdb;
                    #pdb.set_trace()
                    bab_out = 'ET'
        print(f"Nb states visited: {bab_nb_states}")
        # print('bnb takes: ', bnb_time)
        print('\n')

        bab_end = time.time()
        bab_time = bab_end - bab_start
        print('total time required: ', bab_time)

        print('\n')

        if args.record:
            graph_df.loc[new_idx]["Idx"] = imag_idx
            graph_df.loc[new_idx]["Eps"] = eps_temp
            graph_df.loc[new_idx]["prop"] = prop_idx

            if args.branching_choice == "heuristic":
                graph_df.loc[new_idx][f"BSAT_KW_{algorithm_name}"] = bab_out
                graph_df.loc[new_idx][f"BBran_KW_{algorithm_name}"] = bab_nb_states
                graph_df.loc[new_idx][f"BTime_KW_{algorithm_name}"] = bab_time
            else:
                graph_df.loc[new_idx][f"BSAT_GNN_{algorithm_name}"] = bab_out
                graph_df.loc[new_idx][f"BBran_GNN_{algorithm_name}"] = bab_nb_states
                graph_df.loc[new_idx][f"BTime_GNN_{algorithm_name}"] = bab_time
                graph_df.loc[new_idx][f"FS_Ratio_{algorithm_name}"] = bab_fs
            graph_df.to_pickle(record_name)


if __name__ == '__main__':
    main()
