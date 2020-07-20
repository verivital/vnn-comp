import argparse
import os
import torch
import time
import copy
from torch import nn
from plnn.modules import View
from plnn.model import cifar_model_large, cifar_model
from plnn.proxlp_solver.solver import SaddleLP
from plnn.mip_solver import MIPNetwork
from plnn.network_linear_approximation import LinearizedNetwork
from cifar_bound_comparison import load_network, make_elided_models, cifar_loaders
from convex_adversarial import robust_loss
from convex_adversarial.dual_network import RobustBounds, DualNetwork

def main():
    parser = argparse.ArgumentParser(description="Compute and time a bunch of bounds.")
    parser.add_argument('network_filename', type=str,
                        help='Path to the network')
    parser.add_argument('eps', type=float,
                        help='Epsilon - default: 0.0347')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timeout per problem. (in s)')
    parser.add_argument('--modulo', type=int,
                        help='Numbers of a job to split the dataset over.')
    parser.add_argument('--modulo_do', type=int,
                        help='Which job_id is this one.')
    args = parser.parse_args()
    model = load_network(args.network_filename)

    elided_models = make_elided_models(model)

    # Verify the quality of the network
    sanity_check = False
    use_cuda = True
    if sanity_check:
        _, test_loader = cifar_loaders(1)
        err = 0
        rob_err = 0
        seen = 0
        eli_err = 0
        if use_cuda:
            model = model.cuda()
        with torch.no_grad():
            for _, (X, y) in enumerate(test_loader):
                elimodel = elided_models[y.item()]
                if use_cuda:
                    elimodel = elimodel.cuda()
                    X = X.cuda()
                    y = y.cuda()
                out = model(X)
                _, robust_err = robust_loss(model, args.eps, X, y, proj=None, bounded_input=False)
                eli_out = elimodel(X)


                err += (out.max(1)[1] != y).float().sum()
                rob_err += robust_err * out.size(0)
                eli_err += (eli_out.min(1)[0] < 0).float().sum()
                seen += out.size(0)

                print(f"Error  rate: {err/seen}")
                print(f"Elided rate: {eli_err/seen}")
                print(f"Robust error rate: {rob_err/seen}")


    _, test_loader = cifar_loaders(1)
    for idx, (X, y) in enumerate(test_loader):
        if (args.modulo is not None) and (idx % args.modulo != args.modulo_do):
            continue
        elided_model = elided_models[y.item()]
        to_ignore = y.item()

        # domain = torch.stack([X.squeeze(0) - args.eps,
        #                       X.squeeze(0) + args.eps], dim=-1)
        domain = torch.stack([X.squeeze(0) - args.eps,
                              X.squeeze(0) + args.eps], dim=-1)

        mip_net = MIPNetwork([lay for lay in elided_model])
        mip_start = time.time()
        mip_net.setup_model(domain, bounds="interval-kw", use_obj_function=True)
        mip_ready = time.time()
        sat, solution, nb_visited_states = mip_net.solve(domain, timeout=args.timeout)
        mip_end = time.time()
        setup_time = mip_ready - mip_start
        solve_time = mip_end - mip_ready
        if sat:
            print(f"SAT - build: {setup_time} s \t solve: {solve_time} s \t nb states: {nb_visited_states}")
        elif sat is False:
            print(f"UNSAT - build: {setup_time} s \t solve: {solve_time} s \t nb states: {nb_visited_states}")
        elif sat is None:
            print(f"Timeout - build: {setup_time} s \t solve: {solve_time} s \t nb states: {nb_visited_states}")


if __name__ == '__main__':
    main()
