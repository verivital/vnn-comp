import argparse
import torch
from tools.cifar_bound_comparison import load_network, make_elided_models, cifar_loaders
from tools.bab_tools.model_utils import load_cifar_1to1_exp, load_1to1_eth
from plnn.proxlp_solver.solver import SaddleLP
from plnn.proxlp_solver.dj_relaxation import DJRelaxationLP
from plnn.network_linear_approximation import LinearizedNetwork
from plnn.anderson_linear_approximation import AndersonLinearizedNetwork
import copy, csv
import time, os, math
import pandas as pd


def run_lower_bounding():
    parser = argparse.ArgumentParser(description="Compute a bound and plot the results")

    # Argument option 1: pass network filename, epsilon, image index.
    parser.add_argument('--network_filename', type=str, help='Path of the network')
    parser.add_argument('--eps', type=float, help='Epsilon')
    parser.add_argument('--img_idx', type=int, default=0)

    # Argument option 2: pass jodie's rlv name, network, and an index i to use the i-th property in that file
    parser.add_argument('--pdprops', type=str, help='pandas table with all props we are interested in')
    parser.add_argument('--nn_name', type=str, help='network architecture name')
    parser.add_argument('--prop_idx', type=int, default=0)

    parser.add_argument('--data', type=str, default='cifar')
    parser.add_argument('--eta', type=float)
    parser.add_argument('--feta', type=float)
    parser.add_argument('--init_step', type=float)
    parser.add_argument('--fin_step', type=float)
    parser.add_argument('--out_iters', type=int)
    parser.add_argument('--prox_momentum', type=float, default=0)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--n_cuts', type=float, default=100)
    parser.add_argument('--max_solver_batch', type=float, default=10000,
                        help='max batch size for bounding computations')
    parser.add_argument('--define_linear_approximation', action='store_true',
                        help="if this flag is true, compute all intermediate bounds w/ the selected algorithm")
    parser.add_argument('--algorithm', type=str, choices=["planet-adam", "proxlp",
                                                          "gurobi", "gurobi-anderson", "dj-adam"],
                        help="which algorithm to use, in case one does init or uses it alone")
    
    args = parser.parse_args()
    if args.network_filename and args.data == 'cifar':#UAI
        # Load all the required data, setup the model
        model = load_network(args.network_filename)
        elided_models = make_elided_models(model)
        _, test_loader = cifar_loaders(1)
        for idx, (X, y) in enumerate(test_loader):
            if idx != args.img_idx:
                continue
            model = elided_models[y.item()]
        domain = torch.stack([X.squeeze(0) - args.eps,
                              X.squeeze(0) + args.eps], dim=-1).cuda()
    elif args.nn_name:
        if args.data == 'cifar' and args.pdprops != None:#cifar-jodie
            # load all properties
            path = './batch_verification_results/'
            gt_results = pd.read_pickle(path + args.pdprops)
            batch_ids = gt_results.index

            for new_idx, idx in enumerate(batch_ids):
                if idx != args.prop_idx:
                    continue
                imag_idx = gt_results.loc[idx]["Idx"]
                prop_idx = gt_results.loc[idx]['prop']
                eps_temp = gt_results.loc[idx]["Eps"]
                # skip the nan prop_idx or eps_temp (happens in wide.pkl, jodie's mistake, I guess)
                if (math.isnan(imag_idx) or math.isnan(prop_idx) or math.isnan(eps_temp)):
                    continue

                x, model, test = load_cifar_1to1_exp(args.nn_name, imag_idx, prop_idx)
                # since we normalise cifar data set, it is unbounded now
                assert test == prop_idx
                domain = torch.stack([x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp], dim=-1)

        elif args.pdprops == None: #ETH. mnist-eth and cifar-eth
            csvfile = open('././data/%s_test.csv'%(args.data), 'r')
            tests = list(csv.reader(csvfile, delimiter=','))

            eps_temp = float(args.nn_name[6:]) if args.data == 'mnist' else float(args.nn_name.split('_')[1])/float(args.nn_name.split('_')[2])

            x, model, test = load_1to1_eth(args.data, args.nn_name, idx=args.prop_idx, test=tests, eps_temp=eps_temp,
                                           max_solver_batch=args.max_solver_batch)
            # since we normalise cifar data set, it is unbounded now
            prop_idx = test
            domain = torch.stack([x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp], dim=-1)

    lin_approx_string = "" if not args.define_linear_approximation else "-allbounds"
    image = args.prop_idx if args.nn_name else args.img_idx

    # compute intermediate bounds with KW. Use only these for every method to allow comparison on the last layer
    # and optimize only the last layer
    cuda_elided_model = copy.deepcopy(model).cuda() if args.network_filename and args.data == 'cifar' else \
        [copy.deepcopy(lay).cuda() for lay in model]
    cuda_domain = domain.unsqueeze(0).cuda()
    intermediate_net = SaddleLP([lay for lay in cuda_elided_model], max_batch=args.max_solver_batch)
    with torch.no_grad():
        intermediate_net.set_solution_optimizer('best_naive_kw', None)
        intermediate_net.define_linear_approximation(cuda_domain, no_conv=False)
    intermediate_ubs = intermediate_net.upper_bounds
    intermediate_lbs = intermediate_net.lower_bounds

    folder = f"./timings_{args.data}/"
    if not os.path.exists(folder):
        os.makedirs(folder)
    if args.nn_name:
        folder += f"{args.nn_name}_"

    if args.algorithm == "proxlp":
        # ProxLP
        acceleration_dict = {
            'momentum': args.prox_momentum,  # decent momentum: 0.6 w/ increasing eta
        }

        optprox_params = {
            'nb_total_steps': args.out_iters,
            'max_nb_inner_steps': 2,  # this is 2/5 as simpleprox
            'eta': args.eta,  # eta is kept the same as in simpleprox
            'initial_eta': args.eta if args.feta else None,
            'final_eta': args.feta if args.feta else None,
            'log_values': False,
            'inner_cutoff': 0,
            'maintain_primal': True,
            'acceleration_dict': acceleration_dict
        }
        optprox_net = SaddleLP(cuda_elided_model, store_bounds_progress=len(intermediate_net.weights),
                               max_batch=args.max_solver_batch)
        optprox_start = time.time()
        with torch.no_grad():
            optprox_net.set_decomposition('pairs', 'naive')
            optprox_net.set_solution_optimizer('optimized_prox', optprox_params)
            if not args.define_linear_approximation:
                optprox_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                lb, ub = optprox_net.compute_lower_bound()
            else:
                optprox_net.define_linear_approximation(cuda_domain)
                lb = optprox_net.lower_bounds[-1]
                ub = optprox_net.upper_bounds[-1]
        optprox_end = time.time()
        optprox_time = optprox_end - optprox_start
        optprox_lbs = lb.cpu().mean()
        optprox_ubs = ub.cpu().mean()
        print(f"ProxLP Time: {optprox_time}")
        print(f"ProxLP LB: {optprox_lbs}")
        print(f"ProxLP UB: {optprox_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},LB:{optprox_lbs},Time:{optprox_time},Eta{args.eta},Out-iters:{args.out_iters}\n")

        acceleration_string = f"-mom:{args.prox_momentum}"
        pickle_name = folder + f"timings-img{image}-{args.algorithm},eta:{args.eta}-feta:{args.feta}{acceleration_string}{lin_approx_string}.pickle"
        torch.save(optprox_net.logger, pickle_name)

    elif args.algorithm == "planet-adam":
        adam_params = {
            'nb_steps': args.out_iters,
            'initial_step_size': args.init_step,
            'final_step_size': args.fin_step,
            'betas': (args.beta1, 0.999),
            'log_values': False
        }
        adam_net = SaddleLP(cuda_elided_model, store_bounds_progress=len(intermediate_net.weights),
                            max_batch=args.max_solver_batch)
        adam_start = time.time()
        with torch.no_grad():
            adam_net.set_decomposition('pairs', 'naive')
            adam_net.set_solution_optimizer('adam', adam_params)
            if not args.define_linear_approximation:
                adam_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                lb, ub = adam_net.compute_lower_bound()
            else:
                adam_net.define_linear_approximation(cuda_domain)
                lb = adam_net.lower_bounds[-1]
                ub = adam_net.upper_bounds[-1]
        adam_end = time.time()
        adam_time = adam_end - adam_start
        adam_lbs = lb.cpu().mean()
        adam_ubs = ub.cpu().mean()
        print(f"Planet adam Time: {adam_time}")
        print(f"Planet adam LB: {adam_lbs}")
        print(f"Planet adam UB: {adam_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},LB:{adam_lbs},Time:{adam_time},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{image}-{args.algorithm},istepsize:{args.init_step},fstepsize:{args.fin_step},beta1:{args.beta1}{lin_approx_string}.pickle"
        torch.save(adam_net.logger, pickle_name)

    elif args.algorithm == "dj-adam":
        adam_params = {
            'nb_steps': args.out_iters,
            'initial_step_size': args.init_step,
            'final_step_size': args.fin_step,
            'betas': (args.beta1, 0.999),
            'log_values': False
        }
        djadam_net = DJRelaxationLP(cuda_elided_model, params=adam_params,
                                    store_bounds_progress=len(intermediate_net.weights),
                                    max_batch=args.max_solver_batch)
        djadam_start = time.time()
        with torch.no_grad():
            if not args.define_linear_approximation:
                djadam_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                lb, ub = djadam_net.compute_lower_bound()
            else:
                djadam_net.define_linear_approximation(cuda_domain)
                lb = djadam_net.lower_bounds[-1]
                ub = djadam_net.upper_bounds[-1]
        djadam_end = time.time()
        djadam_time = djadam_end - djadam_start
        djadam_lbs = lb.cpu().mean()
        djadam_ubs = ub.cpu().mean()
        print(f"Planet adam Time: {djadam_time}")
        print(f"Planet adam LB: {djadam_lbs}")
        print(f"Planet adam UB: {djadam_ubs}")
        with open("tunings.txt", "a") as file:
            file.write(folder + f"{args.algorithm},LB:{djadam_lbs},Time:{djadam_time},Out-iters:{args.out_iters}\n")
        pickle_name = folder + f"timings-img{image}-{args.algorithm},istepsize:{args.init_step},fstepsize:{args.fin_step},beta1:{args.beta1}{lin_approx_string}.pickle"
        torch.save(djadam_net.logger, pickle_name)

    elif args.algorithm == "gurobi":
        grb_net = LinearizedNetwork([lay for lay in model])
        grb_start = time.time()
        if not args.define_linear_approximation:
            grb_net.build_model_using_bounds(domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs],
                                                      [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]))
            lb, ub = grb_net.compute_lower_bound()
        else:
            grb_net.define_linear_approximation(domain, n_threads=4)
            lb = grb_net.lower_bounds[-1]
            ub = grb_net.upper_bounds[-1]
        grb_end = time.time()
        grb_time = grb_end - grb_start
        print(f"Gurobi Time: {grb_time}")
        print(f"Gurobi LB: {lb}")
        print(f"Gurobi UB: {ub}")
        with open("tunings.txt", "a") as file:
            file.write(f"{args.algorithm},LB:{lb},Time:{grb_time}\n")

    elif args.algorithm == "gurobi-anderson":
        bounds_net = AndersonLinearizedNetwork(
            [lay for lay in model], mode="lp-cut", n_cuts=args.n_cuts, cuts_per_neuron=True)
        grb_start = time.time()
        if not args.define_linear_approximation:
            bounds_net.build_model_using_bounds(domain, ([lbs.cpu().squeeze(0) for lbs in intermediate_lbs],
                                                      [ubs.cpu().squeeze(0) for ubs in intermediate_ubs]))
            lb, ub = bounds_net.compute_lower_bound()
        else:
            bounds_net.define_linear_approximation(domain, n_threads=4)
            lb = bounds_net.lower_bounds[-1]
            ub = bounds_net.upper_bounds[-1]
        grb_end = time.time()
        grb_time = grb_end - grb_start
        print(f"Gurobi Anderson Time: {grb_time}")
        print(f"Gurobi Anderson LB: {lb}")
        print(f"Gurobi Anderson UB: {ub}")
        with open("tunings.txt", "a") as file:
            file.write(f"{args.algorithm},LB:{lb},Time:{grb_time}\n")

    # if lb < 0 and args.nn_name:
    #     with open(f"undecided-{args.data}-eps{eps_temp}.txt", "a") as file:
    #         file.write(f"{args.prop_idx}, ")


if __name__ == '__main__':
    run_lower_bounding()
