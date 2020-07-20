from plnn.branch_and_bound.relu_branch_and_bound import relu_bab
from plnn.proxlp_solver.solver import SaddleLP
from tools.cifar_bound_comparison import load_network, make_elided_models, cifar_loaders
import torch, copy, argparse

# TODO! (use relu_branch_and_bound)

# TODO: all the get_lower_bound and get_upper_bound functions seem to be properly implemented.

# TODO: while it's not going to be easy to batchify the operations, I can still argue with Pawan that computing intermediate bounds would provide a speed-up

def parse_input_load_cifar():
    parser = argparse.ArgumentParser(description="Compute a bound and plot the results")
    parser.add_argument('network_filename', type=str,
                        help='Path ot the network')
    parser.add_argument('eps', type=float, help='Epsilon')
    parser.add_argument('--img_idx', type=int, default=0)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--out_iters', type=int)
    parser.add_argument('--algorithm', type=str,
                        choices=["bigm-prox", "bigm-adam", "planet-adam", "proxlp", "gurobi-time"],
                        help="which algorithm to use, in case one does init or uses it alone")

    args = parser.parse_args()

    # Load all the required data, setup the model
    model = load_network(args.network_filename)
    elided_models = make_elided_models(model)
    _, test_loader = cifar_loaders(1)
    for idx, (X, y) in enumerate(test_loader):
        if idx != args.img_idx:
            continue
        elided_model = elided_models[y.item()]
    domain = torch.stack([X.squeeze(0) - args.eps,
                          X.squeeze(0) + args.eps], dim=-1).cuda()
    return args, domain, elided_model


def runner():

    args, domain, elided_model = parse_input_load_cifar()
    cuda_elided_model = copy.deepcopy(elided_model).cuda()
    cuda_domain = domain.cuda()

    if args.algorithm == "proxlp":
        # ProxLP
        optprox_params = {
            'nb_total_steps': args.out_iters,
            'max_nb_inner_steps': 2,  # this is 2/5 as simpleprox
            'eta': args.eta,  # eta is kept the same as in simpleprox
            'log_values': False,
            'inner_cutoff': 0,
            'maintain_primal': True
        }
        network = SaddleLP([lay for lay in cuda_elided_model])
        network.set_decomposition('pairs', 'KW')
        network.set_solution_optimizer('optimized_prox', optprox_params)

    epsilon = 0
    decision_bound = 0
    # TODO: I need to torch.no_grad when calling lower bounds
    min_lb, min_ub, ub_point, nb_visited_states = relu_bab(network, cuda_domain,
                                                           epsilon, decision_bound)
    if min_lb >= 0:
        print("UNSAT")
    elif min_ub < 0:
        # Verify that it is a valid solution
        candidate_ctx = ub_point.view(1, -1)
        val = network.net(candidate_ctx)
        margin = val.squeeze().item()
        if margin > 0:
            print("Error")
        else:
            print("SAT")
        print(ub_point)
        print(margin)
    else:
        print("Unknown")
    print(f"Nb states visited: {nb_visited_states}")


if __name__ == "__main__":
    runner()