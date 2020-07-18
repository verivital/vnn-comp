import argparse
import os
import torch
import time
import copy
from torch import nn
from plnn.model import cifar_model, cifar_model_large
from plnn.proxlp_solver.solver import SaddleLP
from plnn.naive_approximation import NaiveNetwork
from plnn.network_linear_approximation import LinearizedNetwork
from convex_adversarial import robust_loss
from convex_adversarial.dual_network import RobustBounds, DualNetwork

def load_network(filename):
    dump = torch.load(filename)
    state_dict = dump['state_dict'][0]
    if len(state_dict) == 8:
        model = cifar_model()
    elif len(state_dict) == 14:
        model = cifar_model_large()
    else:
        raise NotImplementedError
    # [0] because it's the dumb cascade training, which we don't deal with
    model.load_state_dict(state_dict)
    return model


def make_elided_models(model, return_error=False):
    """
    Default is to return GT - other
    Set `return_error` to True to get instead something that returns a loss
    (other - GT)

    mono_output=False is an argument I removed
    """
    elided_models = []
    layers = [lay for lay in model]
    assert isinstance(layers[-1], nn.Linear)

    net = layers[:-1]
    last_layer = layers[-1]
    nb_classes = last_layer.out_features

    for gt in range(nb_classes):
        new_layer = nn.Linear(last_layer.in_features,
                              last_layer.out_features-1)

        wrong_weights = last_layer.weight[[f for f in range(last_layer.out_features) if f != gt], :]
        wrong_biases = last_layer.bias[[f for f in range(last_layer.out_features) if f != gt]]

        if return_error:
            new_layer.weight.data.copy_(wrong_weights - last_layer.weight[gt])
            new_layer.bias.data.copy_(wrong_biases - last_layer.bias[gt])
        else:
            new_layer.weight.data.copy_(last_layer.weight[gt] - wrong_weights)
            new_layer.bias.data.copy_(last_layer.bias[gt] - wrong_biases)

        layers = copy.deepcopy(net) + [new_layer]
        # if mono_output and new_layer.out_features != 1:
        #     layers.append(View((1, new_layer.out_features)))
        #     layers.append(nn.MaxPool1d(new_layer.out_features,
        #                                stride=1))
        #     layers.append(View((1,)))
        new_elided_model = nn.Sequential(*layers)
        elided_models.append(new_elided_model)
    return elided_models


def cifar_loaders(batch_size):
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.225, 0.225, 0.225])
    train = datasets.CIFAR10('./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]))
    test = datasets.CIFAR10('./data', train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]))
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
        shuffle=False, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size,
        shuffle=False, pin_memory=True)
    return train_loader, test_loader


def dump_bounds(target_file, time, upper_bounds, to_ignore=None):
    bounds_list = upper_bounds.squeeze().numpy().tolist()
    if to_ignore is not None:
        # There is one of the optimization that is unnecessary: the one with
        # robustness to the ground truth.
        del bounds_list[to_ignore]
    bound_str = "\t".join(map(str, bounds_list))
    with open(target_file, 'w') as res_file:
        res_file.write(f"{time}\n{bound_str}\n")

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
    args = parser.parse_args()
    model = load_network(args.network_filename)

    results_dir = args.target_directory
    os.makedirs(results_dir, exist_ok=True)

    elided_models = make_elided_models(model, True)

    # Verify the quality of the network
    sanity_check = False
    if sanity_check:
        _, test_loader = cifar_loaders(8)
        err = 0
        rob_err = 0
        seen = 0
        for _, (X, y) in enumerate(test_loader):
            out = model(X)
            _, robust_err = robust_loss(model, args.eps, X, y, proj=None, bounded_input=False)
            err += (out.max(1)[1] != y).float().sum()
            rob_err += robust_err * out.size(0)
            seen += out.size(0)

        print(f"Robust error rate: {rob_err/seen}")
        print(f"Robust error rate: {rob_err/seen}")


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

        ## Naive bounds
        naive_target_file = os.path.join(target_dir, "Naive.txt")
        if not os.path.exists(naive_target_file):
            naive_net = NaiveNetwork([lay for lay in elided_model])
            naive_start = time.time()
            with torch.no_grad():
                naive_net.do_interval_analysis(domain)
            naive_end = time.time()
            naive_time = naive_end - naive_start
            naive_ubs = naive_net.upper_bounds[-1]
            dump_bounds(naive_target_file, naive_time, naive_ubs)

        ## Kolter & Wong bounds
        kw_target_file = os.path.join(target_dir, "KW.txt")
        if not os.path.exists(kw_target_file):
            cuda_model = copy.deepcopy(model).cuda()
            cuda_X = X.detach().cuda()
            cuda_y = y.detach().cuda()
            rb_kw_start = time.time()
            with torch.no_grad():
                rb_kw_ubs = RobustBounds(cuda_model, args.eps)(cuda_X, cuda_y)
            rb_kw_end = time.time()
            rb_kw_time = rb_kw_end - rb_kw_start
            rb_kw_ubs = rb_kw_ubs.cpu()
            dump_bounds(kw_target_file, rb_kw_time, rb_kw_ubs, to_ignore)

        ## Proximal methods
        for simpleprox_steps in [100, 500, 1000]:
            simpleprox_params = {
                'nb_total_steps': simpleprox_steps,
                'max_nb_inner_steps': 5,
                'eta': 1e3,
                'log_values': False,
                'inner_cutoff': 0,
                'maintain_primal': True
            }
            simpleprox_target_file = os.path.join(target_dir, f"Proximal_{simpleprox_steps}.txt")
            gurobi_samebudget_target_file = os.path.join(target_dir, f"Gurobi_anytime-{simpleprox_steps}steps_equivalent.txt")
            if not os.path.exists(simpleprox_target_file):
                cuda_elided_model = copy.deepcopy(elided_model).cuda()
                cuda_domain = domain.cuda()
                simpleprox_net = SaddleLP([lay for lay in cuda_elided_model])
                simpleprox_start = time.time()
                with torch.no_grad():
                    simpleprox_net.set_decomposition('pairs', 'KW')
                    simpleprox_net.set_solution_optimizer('prox', simpleprox_params)
                    simpleprox_net.define_linear_approximation(cuda_domain, force_optim=False, no_conv=False)
                simpleprox_end = time.time()
                simpleprox_time = simpleprox_end - simpleprox_start
                simpleprox_ubs = simpleprox_net.upper_bounds[-1].cpu()

                # TODO: try increasing n_threads to 3
                grbeq_net = LinearizedNetwork([lay for lay in elided_model])
                grbeq_start = time.time()
                grbeq_net.define_linear_approximation(domain, force_optim=False,
                                                      time_limit_per_layer=simpleprox_net.opt_time_per_layer)
                grbeq_end = time.time()
                grbeq_time = grbeq_end - grbeq_start
                grbeq_ubs = torch.Tensor(grbeq_net.upper_bounds[-1])

                del simpleprox_net
                dump_bounds(simpleprox_target_file, simpleprox_time, simpleprox_ubs)
                del grbeq_net
                dump_bounds(gurobi_samebudget_target_file, grbeq_time, grbeq_ubs)

        ## Proximal methods
        for optprox_steps in [40, 200, 400]:  # This is set to generate the same number of outer steps as simpleprox.
            optprox_params = {
                'nb_total_steps': optprox_steps,
                'max_nb_inner_steps': 2,  # this is 2/5 as simpleprox
                'eta': 1e3,  # eta is kept the same as in simpleprox
                'log_values': False,
                'inner_cutoff': 0,
                'maintain_primal': True
            }
            optprox_target_file = os.path.join(target_dir, f"Optproximal_{optprox_steps}.txt")
            if not os.path.exists(optprox_target_file):
                cuda_elided_model = copy.deepcopy(elided_model).cuda()
                cuda_domain = domain.cuda()
                optprox_net = SaddleLP([lay for lay in cuda_elided_model])
                optprox_start = time.time()
                with torch.no_grad():
                    optprox_net.set_decomposition('pairs', 'KW')
                    optprox_net.set_solution_optimizer('optimized_prox', optprox_params)
                    optprox_net.define_linear_approximation(cuda_domain, force_optim=False, no_conv=False)
                optprox_end = time.time()
                optprox_time = optprox_end - optprox_start
                optprox_ubs = optprox_net.upper_bounds[-1].cpu()
                del optprox_net
                dump_bounds(optprox_target_file, optprox_time, optprox_ubs)

        ## Proximal decay methods
        for simpleprox_steps in [100, 500, 1000]:
            simpleprox_params = {
                'nb_total_steps': simpleprox_steps,
                'max_nb_inner_steps': 5,
                'initial_eta': 1e3,
                'final_eta': 1,
                'log_values': False,
                'inner_cutoff': 0,
                'maintain_primal': True
            }
            simpleprox_target_file = os.path.join(target_dir, f"Proximal_etaschedule_{simpleprox_steps}.txt")
            if not os.path.exists(simpleprox_target_file):
                cuda_elided_model = copy.deepcopy(elided_model).cuda()
                cuda_domain = domain.cuda()
                simpleprox_net = SaddleLP([lay for lay in cuda_elided_model])
                simpleprox_start = time.time()
                with torch.no_grad():
                    simpleprox_net.set_decomposition('pairs', 'KW')
                    simpleprox_net.set_solution_optimizer('prox', simpleprox_params)
                    simpleprox_net.define_linear_approximation(cuda_domain, force_optim=False, no_conv=False)
                simpleprox_end = time.time()
                simpleprox_time = simpleprox_end - simpleprox_start
                simpleprox_ubs = simpleprox_net.upper_bounds[-1].cpu()
                del simpleprox_net
                dump_bounds(simpleprox_target_file, simpleprox_time, simpleprox_ubs)

        # TODO: remove simple prox and use only optprox (the update can be improved)
        ## Optimised proximal decay methods
        for optprox_steps in [40, 200, 400]:  # This is set to generate the same number of outer steps as simpleprox.
            optprox_params = {
                'nb_total_steps': optprox_steps,
                'max_nb_inner_steps': 2,  # this is 2/5 as simpleprox
                'initial_eta': 1e3,  # eta is kept the same as in simpleprox
                'final_eta': 1,
                'log_values': False,
                'inner_cutoff': 0,
                'maintain_primal': True
            }
            optprox_target_file = os.path.join(target_dir, f"Optproximal_etaschedule_{optprox_steps}.txt")
            if not os.path.exists(optprox_target_file):
                cuda_elided_model = copy.deepcopy(elided_model).cuda()
                cuda_domain = domain.cuda()
                optprox_net = SaddleLP([lay for lay in cuda_elided_model])
                optprox_start = time.time()
                with torch.no_grad():
                    optprox_net.set_decomposition('pairs', 'KW')
                    optprox_net.set_solution_optimizer('optimized_prox', optprox_params)
                    optprox_net.define_linear_approximation(cuda_domain, force_optim=False, no_conv=False)
                optprox_end = time.time()
                optprox_time = optprox_end - optprox_start
                optprox_ubs = optprox_net.upper_bounds[-1].cpu()
                del optprox_net
                dump_bounds(optprox_target_file, optprox_time, optprox_ubs)

        ## Gurobi Bounds
        # TODO: while the optimization methods do not have a 2x overhead in computing the lower bound (batch parallelism), this does.
        # TODO: but once you interpret it as a cpu reference rather than a baseline (as the baseline is subgradient ~DJ-like)
        # TODO: it becomes negligible
        # TODO: try increasing n_threads, though (to 3)
        grb_target_file = os.path.join(target_dir, "Gurobi.txt")
        if not os.path.exists(grb_target_file):
            grb_net = LinearizedNetwork([lay for lay in elided_model])
            grb_start = time.time()
            grb_net.define_linear_approximation(domain, force_optim=False)
            grb_end = time.time()
            grb_time = grb_end - grb_start
            grb_ubs = torch.Tensor(grb_net.upper_bounds[-1])
            dump_bounds(grb_target_file, grb_time, grb_ubs)

        ## ADAM bounds
        for adam_steps in [100, 500, 1000]:
            adam_params = {
                'nb_steps': adam_steps,
                'initial_step_size': 1e-3,
                'final_step_size': 1e-6,
                'betas': (0.9, 0.999),
                'log_values': False
            }
            adam_target_file = os.path.join(target_dir, f"Adam_{adam_steps}.txt")
            if not os.path.exists(adam_target_file):
                cuda_elided_model = copy.deepcopy(elided_model).cuda()
                cuda_domain = domain.cuda()
                adam_net = SaddleLP([lay for lay in cuda_elided_model])
                adam_start = time.time()
                with torch.no_grad():
                    adam_net.set_decomposition('pairs', 'KW')
                    adam_net.set_solution_optimizer('adam', adam_params)
                    adam_net.define_linear_approximation(cuda_domain, force_optim=True, no_conv=False)
                adam_end = time.time()
                adam_time = adam_end - adam_start
                adam_ubs = adam_net.upper_bounds[-1].cpu()
                del adam_net
                dump_bounds(adam_target_file, adam_time, adam_ubs)


if __name__ == '__main__':
    main()
