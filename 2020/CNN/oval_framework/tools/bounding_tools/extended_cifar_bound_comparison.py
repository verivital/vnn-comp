import argparse
import os
import torch
import time
import copy
from torch import nn
from plnn.model import cifar_model, cifar_model_large
from plnn.proxlp_solver.solver import SaddleLP
from plnn.proxlp_solver.dj_relaxation import DJRelaxationLP
from plnn.naive_approximation import NaiveNetwork
from plnn.network_linear_approximation import LinearizedNetwork
from tools.cifar_bound_comparison import load_network, make_elided_models, cifar_loaders, dump_bounds


def main():
    parser = argparse.ArgumentParser(description="Compute and time a bunch of bounds.")
    parser.add_argument('network_filename', type=str,
                        help='Path to the network')
    parser.add_argument('eps', type=float,
                        help='Epsilon - default: 0.0347')
    parser.add_argument('target_directory', type=str,
                        help='Where to store the results')
    parser.add_argument('--modulo', type=int,
                        help='Numbers of a job to split the dataset over.')
    parser.add_argument('--modulo_do', type=int,
                        help='Which job_id is this one.')
    parser.add_argument('--from_intermediate_bounds', action='store_true',
                        help="if this flag is true, intermediate bounds are computed w/ best of naive-KW")
    args = parser.parse_args()
    model = load_network(args.network_filename)

    results_dir = args.target_directory
    os.makedirs(results_dir, exist_ok=True)

    elided_models = make_elided_models(model, True)

    _, test_loader = cifar_loaders(1)
    for idx, (X, y) in enumerate(test_loader):
        if (args.modulo is not None) and (idx % args.modulo != args.modulo_do):
            continue
        target_dir = os.path.join(results_dir, f"{idx}")
        os.makedirs(target_dir, exist_ok=True)
        elided_model = elided_models[y.item()]
        to_ignore = y.item()

        domain = torch.stack([X.squeeze(0) - args.eps,
                              X.squeeze(0) + args.eps], dim=-1)

        lin_approx_string = "" if not args.from_intermediate_bounds else "-fromintermediate"

        # compute intermediate bounds with KW. Use only these for every method to allow comparison on the last layer
        # and optimize only the last layer
        if args.from_intermediate_bounds:
            cuda_elided_model = copy.deepcopy(elided_model).cuda()
            cuda_domain = domain.cuda()
            intermediate_net = SaddleLP([lay for lay in cuda_elided_model])
            with torch.no_grad():
                intermediate_net.set_solution_optimizer('best_naive_kw', None)
                intermediate_net.define_linear_approximation(cuda_domain, force_optim=True, no_conv=False,
                                                             override_numerical_errors=True)
            intermediate_ubs = intermediate_net.upper_bounds
            intermediate_lbs = intermediate_net.lower_bounds

        ## Kolter & Wong bounds
        kw_target_file = os.path.join(target_dir, f"KW{lin_approx_string}.txt")
        if not os.path.exists(kw_target_file):
            cuda_elided_model = copy.deepcopy(elided_model).cuda()
            cuda_domain = domain.cuda()
            kw_net = SaddleLP([lay for lay in cuda_elided_model])
            kw_start = time.time()
            with torch.no_grad():
                kw_net.set_decomposition('pairs', 'KW')
                kw_net.set_solution_optimizer('init', None)
                if not args.from_intermediate_bounds:
                    kw_net.define_linear_approximation(cuda_domain, force_optim=True, no_conv=False)
                    ub = kw_net.upper_bounds[-1]
                else:
                    kw_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                    _, ub = kw_net.compute_lower_bound(all_optim=True)
            kw_end = time.time()
            kw_time = kw_end - kw_start
            kw_ubs = ub.cpu()
            del kw_net
            dump_bounds(kw_target_file, kw_time, kw_ubs)

        ## Proximal methods
        for optprox_steps in [100, 200, 400]:
            optprox_params = {
                'nb_total_steps': optprox_steps,
                'max_nb_inner_steps': 2,  # this is 2/5 as simpleprox
                'initial_eta': 1e1,
                'final_eta': 5e2,
                'log_values': False,
                'inner_cutoff': 0,
                'maintain_primal': True,
                'acceleration_dict': {
                    'momentum': 0.3,  # decent momentum: 0.9 w/ increasing eta
                }
            }
            optprox_target_file = os.path.join(target_dir, f"Proximal_finalmomentum_{optprox_steps}{lin_approx_string}.txt")
            if not os.path.exists(optprox_target_file):
                cuda_elided_model = copy.deepcopy(elided_model).cuda()
                cuda_domain = domain.cuda()
                optprox_net = SaddleLP([lay for lay in cuda_elided_model])
                optprox_start = time.time()
                with torch.no_grad():
                    optprox_net.set_decomposition('pairs', 'KW')
                    optprox_net.set_solution_optimizer('optimized_prox', optprox_params)
                    if not args.from_intermediate_bounds:
                        optprox_net.define_linear_approximation(cuda_domain, force_optim=True, no_conv=False)
                        ub = optprox_net.upper_bounds[-1]
                    else:
                        optprox_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                        _, ub = optprox_net.compute_lower_bound(all_optim=True)
                optprox_end = time.time()
                optprox_time = optprox_end - optprox_start
                optprox_ubs = ub.cpu()

                del optprox_net
                dump_bounds(optprox_target_file, optprox_time, optprox_ubs)

        # Time-limited Gurobi bounds
        optprox_steps = 400
        gurobi_samebudget_target_file = os.path.join(
            target_dir, f"Gurobi_anytime-{optprox_steps}steps_equivalent{lin_approx_string}-fixed.txt")
        if not os.path.exists(gurobi_samebudget_target_file):
            # this will take longer to execute than the one above, as Gurobi takes a certain time to run the first iteration.
            grbeq_net = LinearizedNetwork([lay for lay in elided_model])
            grbeq_start = time.time()
            if not args.from_intermediate_bounds:
                grbeq_net.define_linear_approximation(domain, force_optim=True,
                                                      time_limit_per_layer=optprox_net.opt_time_per_layer, n_threads=4)
                ub = grbeq_net.upper_bounds[-1]
            else:
                grbeq_net.build_model_using_bounds(
                    domain, ([lbs.cpu() for lbs in intermediate_lbs], [ubs.cpu() for ubs in intermediate_ubs]),
                    n_threads=4)
                _, ub = grbeq_net.compute_lower_bound(time_limit_per_layer=[2.], ub_only=True)
                # TODO: time_limit_per_layer is hardcoded, but it reflects the actual average prox time (doesn't make a difference, in practice)
            grbeq_end = time.time()
            grbeq_time = grbeq_end - grbeq_start
            grbeq_ubs = torch.Tensor(ub).cpu()
            del grbeq_net
            dump_bounds(gurobi_samebudget_target_file, grbeq_time, grbeq_ubs)

        ## Gurobi Bounds
        grb_target_file = os.path.join(target_dir, f"Gurobi{lin_approx_string}-fixed.txt")
        if not os.path.exists(grb_target_file):
            grb_net = LinearizedNetwork([lay for lay in elided_model])
            grb_start = time.time()
            if not args.from_intermediate_bounds:
                grb_net.define_linear_approximation(domain, force_optim=True, n_threads=4)
                ub = grb_net.upper_bounds[-1]
            else:
                grb_net.build_model_using_bounds(domain, ([lbs.cpu() for lbs in intermediate_lbs],
                                                          [ubs.cpu() for ubs in intermediate_ubs]), n_threads=4)
                _, ub = grb_net.compute_lower_bound(ub_only=True)
            grb_end = time.time()
            grb_time = grb_end - grb_start
            grb_ubs = torch.Tensor(ub).cpu()
            dump_bounds(grb_target_file, grb_time, grb_ubs)

        ## ADAM bounds
        for adam_steps in [160, 320, 640]:
            adam_params = {
                'nb_steps': adam_steps,
                'initial_step_size': 1e-2,
                'final_step_size': 1e-4,
                'betas': (0.9, 0.999),
                'log_values': False
            }
            adam_target_file = os.path.join(target_dir, f"Adam_fixed_{adam_steps}{lin_approx_string}.txt")
            if not os.path.exists(adam_target_file):
                cuda_elided_model = copy.deepcopy(elided_model).cuda()
                cuda_domain = domain.cuda()
                adam_net = SaddleLP([lay for lay in cuda_elided_model])
                adam_start = time.time()
                with torch.no_grad():
                    adam_net.set_decomposition('pairs', 'KW')
                    adam_net.set_solution_optimizer('adam', adam_params)
                    if not args.from_intermediate_bounds:
                        adam_net.define_linear_approximation(cuda_domain, force_optim=True, no_conv=False)
                        ub = adam_net.upper_bounds[-1]
                    else:
                        adam_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                        _, ub = adam_net.compute_lower_bound(all_optim=True)
                adam_end = time.time()
                adam_time = adam_end - adam_start
                adam_ubs = ub.cpu()
                del adam_net
                dump_bounds(adam_target_file, adam_time, adam_ubs)

        ## DJ-ADAM bounds
        for adam_steps in [260, 520, 1040]:
            adam_params = {
                'nb_steps': adam_steps,
                'initial_step_size': 1e-2,
                'final_step_size': 1e-4,
                'betas': (0.9, 0.999),
                'log_values': False
            }
            djadam_target_file = os.path.join(target_dir, f"DJ_Adam_{adam_steps}{lin_approx_string}.txt")
            if not os.path.exists(djadam_target_file):
                cuda_elided_model = copy.deepcopy(elided_model).cuda()
                cuda_domain = domain.cuda()
                djadam_net = DJRelaxationLP([lay for lay in cuda_elided_model], params=adam_params)
                djadam_start = time.time()
                with torch.no_grad():
                    if not args.from_intermediate_bounds:
                        djadam_net.define_linear_approximation(cuda_domain, force_optim=True, no_conv=False)
                        ub = djadam_net.upper_bounds[-1]
                    else:
                        djadam_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                        _, ub = djadam_net.compute_lower_bound(all_optim=True)
                djadam_end = time.time()
                djadam_time = djadam_end - djadam_start
                djadam_ubs = ub.cpu()
                del djadam_net
                dump_bounds(djadam_target_file, djadam_time, djadam_ubs)

        bigm_methods = False  # ruled out from the ICML20 submission
        if bigm_methods:
            # # Big-M proximal method
            for prox_steps in [10, 30, 60]:  # This is set to very small numbers as it is slow and it underperforms.
                bigm_prox_params = {
                    "initial_eta": 2e2,
                    "nb_inner_iter": 5,
                    "nb_outer_iter": prox_steps,
                    "bigm_algorithm": "prox"
                }
                bigmprox_target_file = os.path.join(target_dir, f"Bigm_prox_{prox_steps}{lin_approx_string}.txt")
                if not os.path.exists(bigmprox_target_file):
                    cuda_elided_model = copy.deepcopy(elided_model).cuda()
                    cuda_domain = domain.cuda()
                    bigmprox_net = ExpLP([lay for lay in cuda_elided_model], params=bigm_prox_params, gurobi_debug=False, debug=False,
                                    bigm="only")
                    bigmprox_start = time.time()
                    with torch.no_grad():
                        if not args.from_intermediate_bounds:
                            bigmprox_net.define_linear_approximation(cuda_domain, force_optim=True, no_conv=False)
                            ub = bigmprox_net.upper_bounds[-1]
                        else:
                            bigmprox_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                            _, ub = bigmprox_net.compute_lower_bound(all_optim=True)
                    bigmprox_end = time.time()
                    bigmprox_time = bigmprox_end - bigmprox_start
                    optprox_ubs = ub.cpu()
                    del bigmprox_net
                    dump_bounds(bigmprox_target_file, bigmprox_time, optprox_ubs)

            # Big-M adam
            for adam_steps in [60, 300, 600]:   # This is set to the same runtime as subgradient on proxlp
                bigm_adam_params = {
                    "nb_outer_iter": adam_steps,
                    "bigm_algorithm": "adam"
                }
                bigmadam_target_file = os.path.join(target_dir, f"Bigm_adam_{adam_steps}{lin_approx_string}.txt")
                if not os.path.exists(bigmadam_target_file):
                    cuda_elided_model = copy.deepcopy(elided_model).cuda()
                    cuda_domain = domain.cuda()
                    bigmadam_net = ExpLP([lay for lay in cuda_elided_model], params=bigm_adam_params,
                                         gurobi_debug=False,
                                         debug=False,
                                         bigm="only")
                    bigmadam_start = time.time()
                    with torch.no_grad():
                        if not args.from_intermediate_bounds:
                            bigmadam_net.define_linear_approximation(cuda_domain, force_optim=True, no_conv=False)
                            ub = bigmadam_net.upper_bounds[-1]
                        else:
                            bigmadam_net.build_model_using_bounds(cuda_domain, (intermediate_lbs, intermediate_ubs))
                            _, ub = bigmadam_net.compute_lower_bound(all_optim=True)
                    bigmadam_end = time.time()
                    bigmadam_time = bigmadam_end - bigmadam_start
                    bigmadam_ubs = ub.cpu()
                    del bigmadam_net
                    dump_bounds(bigmadam_target_file, bigmadam_time, bigmadam_ubs)


if __name__ == '__main__':
    main()
