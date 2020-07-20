import numpy as np
import random

from plnn.branch_and_bound.relu_branch_and_bound import *
import plnn.branch_and_bound.utils as bab
from plnn.branch_and_bound.branching_scores import BranchingChoice
from plnn.branch_and_bound.dumping_utils import dump_branch
import time
from math import floor, ceil



'''
file for generating training datasets
'''
dom_path = './cifar_kw_prox_m2_train_data/'


def relu_traingen(dump_trace, intermediate_net, bounds_net, domain, decision_bound, eps=1e-4, sparsest_layer=0,
             timeout=float("inf"), parent_init_flag=True, batch_max_size=150,
             gurobi_specs=None, gt_throughout = False, total_branches=30):
    '''
    Uses branch and bound algorithm to evaluate the global minimum
    of a given neural network. Splits according to KW.
    Does ReLU activation splitting (not domain splitting, the domain will remain the same throughout)

    Assumes that the last layer is a single neuron.

    `intermediate_net`: Neural Network class, defining the `get_upper_bound`, `define_linear_approximation` functions.
                        Network used to get intermediate bounds.
    `bounds_net`      : Neural Network class, defining the `get_upper_bound`, `define_linear_approximation` functions.
                        Network used to get the final layer bounds, given the intermediate ones.
    `eps`           : Maximum difference between the UB and LB over the minimum
                      before we consider having converged
    `decision_bound`: If not None, stop the search if the UB and LB are both
                      superior or both inferior to this value.
    `batch_size`: The number of domain lower/upper bounds computations done in parallel at once (on a GPU) is
                    batch_size*2
    `max_mem_consumption`: max mem consumption is how much a layer's tensors should occupy, expressed in MB
                    IMPORTANT: high batch_size (>500) has memory issues, will crash.
    `parent_init_flag`: whether to initialize every optimization from its parent node
    `gurobi_specs`: dictionary containing whether ("gurobi") gurobi needs to be used (executes on "p" cpu)
    Returns         : Lower bound and Upper bound on the global minimum,
                      as well as the point where the upper bound is achieved
    '''
    nb_visited_states = 0
    start_time = time.time()

    if gurobi_specs:
        gurobi_dict = dict(gurobi_specs)
        p = gurobi_dict["p"]
        gurobi = gurobi_dict["gurobi"]
    else:
        p = 1
        gurobi = False
    if gurobi and p > 1:
        cpu_servers, server_queue, instruction_queue, barrier = bab.spawn_cpu_servers(p, bounds_net)
        gurobi_dict.update({'server_queue': server_queue, 'instruction_queue': instruction_queue,
                            'barrier': barrier, 'cpu_servers': cpu_servers})
    else:
        gurobi_dict.update({'server_queue': None, 'instruction_queue': None, 'barrier': None, 'cpu_servers': None})

    # do initial computation for the network as it is (batch of size 1: there is only one domain)
    # get intermediate bounds
    intermediate_net.define_linear_approximation(domain.unsqueeze(0))
    intermediate_lbs = copy.deepcopy(intermediate_net.lower_bounds)
    intermediate_ubs = copy.deepcopy(intermediate_net.upper_bounds)

    # compute last layer bounds with a more expensive network
    if not gurobi:
        bounds_net.build_model_using_bounds(domain.unsqueeze(0), (intermediate_lbs, intermediate_ubs))
    else:
        cpu_domain, cpu_intermediate_lbs, cpu_intermediate_ubs = bab.subproblems_to_cpu(
            domain, intermediate_lbs, intermediate_ubs, squeeze_interm=True)
        bounds_net.build_model_using_bounds(cpu_domain, (cpu_intermediate_lbs, cpu_intermediate_ubs))
    global_lb, global_ub = bounds_net.compute_lower_bound(counterexample_verification=True)
    intermediate_lbs[-1] = global_lb
    intermediate_ubs[-1] = global_ub
    bounds_net_device = global_lb.device
    intermediate_net_device = domain.device

    # retrieve bounds info from the bounds network
    global_ub_point = bounds_net.get_lower_bound_network_input()
    global_ub = bounds_net.net(global_ub_point)

    # retrieve which relus are active/passing/ambiguous
    bounds_net.relu_mask = [c_mask.to(bounds_net_device) for c_mask in intermediate_net.relu_mask]
    updated_mask = intermediate_net.relu_mask
    parent_init = bounds_net.last_duals

    print(f"Global LB: {global_lb}; Global UB: {global_ub}")
    if global_lb > decision_bound or global_ub < decision_bound:
        bab.join_children(gurobi_dict, timeout)
        return global_lb, global_ub, global_ub_point, nb_visited_states

    candidate_domain = ReLUDomain(updated_mask, global_lb, global_ub, intermediate_lbs, intermediate_ubs, parent_init,
                                  global_ub_point).to_cpu()
    domains = [candidate_domain]

    branching_tools = BranchingChoice(updated_mask, sparsest_layer, intermediate_net.weights, None)

    infeasible_count = 0
    
    ## MAIN LOOP
    #import pdb; pdb.set_trace()
    # When throughout is false, we only generate #total_branches for 
    # each property 
    if not gt_throughout:
        steps_required = random.randint(1,15)
        step_number = 0
        branch_number = 0
        print(f'generate only {total_branches} train data for the property')
    else:
        print('generate train data throughout for the property')

    while global_ub - global_lb > eps:

        # Check if we have run out of time.
        if time.time() - start_time > timeout:
            bab.join_children(gurobi_dict, timeout)
            return None, None, None, nb_visited_states

        #for batch_idx in range(effective_batch_size):
            # Pick a domain to branch over and remove that from our current list of
            # domains. Also, potentially perform some pruning on the way.
        candidate_domain = bab.pick_out(domains, global_ub.cpu() - eps).to_device(intermediate_net_device)
        # Generate new, smaller domains by splitting over a ReLU
        mask = candidate_domain.mask
        orig_lbs = candidate_domain.lower_all
        orig_ubs = candidate_domain.upper_all
        init_branching_choices, scores = branching_tools.heuristic_branching_decision([orig_lbs], [orig_ubs], [mask])
        score = [i[0] for i in scores]
        lin_mask = [i.view(-1) for i in mask]
        if gt_throughout:
            selected_branching_choices = testing_indices(lin_mask, score)
            #selected_branching_choices = [[1, 804]]*100
            dump = True
            branch_name = dom_path + dump_trace +'_minsum_branch_{}'.format(nb_visited_states)
        else:
            if step_number == steps_required:
                selected_branching_choices = testing_indices(lin_mask, score)
                branch_number += 1
                step_number = 0
                steps_required =random.randint(1,15)
                dump = True
                branch_name = dom_path + dump_trace +'_minsum_fakebranch_{}'.format(nb_visited_states)
            else:
                selected_branching_choices = init_branching_decision
                dump = False
                step_number += 1

        print('the list of selected branching choices')
        print(selected_branching_choices)
        print(f'total branching choices: {len(selected_branching_choices)}')
        #import pdb; pdb.set_trace()
        final_branching_decision_index = None
        current_best_branching_score = 0
        gt_lb_relu = {}

        sub_batches = ceil(len(selected_branching_choices)/batch_max_size)
        for sub_idx in range(sub_batches):
            branching_layer_log = []
            # List of sets storing for each layer index, the batch entries that are splitting a ReLU there
            for _ in range(len(intermediate_net.lower_bounds)-1):
                branching_layer_log.append(set())
            sub_start = sub_idx*batch_max_size
            sub_end = min((sub_idx+1)*batch_max_size, len(selected_branching_choices))
            sub_selected_branching_choices = selected_branching_choices[sub_start: sub_end]
            effective_batch_size = len(sub_selected_branching_choices)
            print(f"effective_batch_size {effective_batch_size}")

            # effective_batch_size*2 as every candidate domain is split in two different ways
            splitted_lbs_stacks = [lbs[0].to(intermediate_net_device).unsqueeze(0).repeat(((effective_batch_size*2,) + (1,) * (lbs.dim() - 1)))
                                   for lbs in bounds_net.lower_bounds]
            splitted_ubs_stacks = [ubs[0].to(intermediate_net_device).unsqueeze(0).repeat(((effective_batch_size*2,) + (1,) * (ubs.dim() - 1)))
                                   for ubs in bounds_net.upper_bounds]
            splitted_domain = domain.unsqueeze(0).expand(((effective_batch_size*2,) + (-1,) * domain.dim()))


            # If SaddleLP/Gurobi, the parent init is stored as a list of tensors for rho
        
            parent_init_stacks = [pinits[0].clone().unsqueeze(0).repeat(((effective_batch_size*2,) + (1,) * (pinits.dim() - 1)))
                                  for pinits in candidate_domain.parent_solution]

            for batch_idx in range(effective_batch_size):
                # get the right branching decision
                branching_decision = sub_selected_branching_choices[batch_idx]
                branching_layer_log[branching_decision[0]] |= {2 * batch_idx, 2 * batch_idx + 1}

                for choice in [0, 1]:
                    #print(f'splitting decision: {branching_decision} - choice {choice}')
                    ## Find the upper and lower bounds on the minimum in the domain
                    ## defined by n_mask_i
                    #if (nb_visited_states % 10) == 0:
                    #    print(f"Running Nb states visited: {nb_visited_states}")
                    #    print(f"N. infeasible nodes {infeasible_count}")

                    # split the domain with the current branching decision
                    splitted_lbs_stacks, splitted_ubs_stacks = update_bounds_from_split(
                        branching_decision, choice, orig_lbs, orig_ubs, 2*batch_idx + choice, splitted_lbs_stacks,
                        splitted_ubs_stacks)

            #import pdb; pdb.set_trace()
            relu_start = time.time()
            # compute the bounds on the batch of splits, at once
            sub_dom_ub, sub_dom_lb, sub_dom_ub_point, sub_updated_mask, sub_dom_lb_all, sub_dom_ub_all, sub_dual_solutions = compute_bounds(
                intermediate_net, bounds_net, branching_layer_log, splitted_domain, splitted_lbs_stacks, splitted_ubs_stacks,
                parent_init_stacks, parent_init_flag, gurobi_dict
            )

            # sorting and comparing all potential choices for a single branch
            final_branching_decision_index = 0
            for idx in range(effective_batch_size):
                layer, index = sub_selected_branching_choices[idx]
                lbs = sub_dom_lb[2*idx:2*idx+2].cpu()
                # record the computed lower bound
                try:
                    gt_lb_relu[layer][index] = lbs
                except KeyError:
                    gt_lb_relu[layer]={}
                    gt_lb_relu[layer][index] = lbs
                # evaluate the quality of decision
                lb_relu_score = min(0, lbs[0]) + min(0, lbs[1]) - 2*global_lb.cpu()
                if lb_relu_score >= current_best_branching_score:
                    final_branching_decision_index = idx
                    current_best_branching_score = lb_relu_score
            final_branching_decision = sub_selected_branching_choices[final_branching_decision_index]
            # remove information not required
            dom_ub = sub_dom_ub[[2 * final_branching_decision_index, 2 * final_branching_decision_index + 1]]
            dom_lb = sub_dom_lb[[2 * final_branching_decision_index, 2 * final_branching_decision_index + 1]]
            dom_ub_point = sub_dom_ub_point[
                [2 * final_branching_decision_index, 2 * final_branching_decision_index + 1]]
            updated_mask = [i[[2 * final_branching_decision_index, 2 * final_branching_decision_index + 1]] for i in sub_updated_mask]
            dom_lb_all = [i[[2 * final_branching_decision_index, 2 * final_branching_decision_index + 1]] for i in sub_dom_lb_all]
            dom_ub_all = [i[[2 * final_branching_decision_index, 2 * final_branching_decision_index + 1]] for i in sub_dom_ub_all]
            dual_solutions = [i[[2 * final_branching_decision_index, 2 * final_branching_decision_index + 1]] for i in sub_dual_solutions]

        # import pdb; pdb.set_trace()
        print(f'final branching decision {final_branching_decision}')
        # dump the finished domain as a training branch            
        if dump:
            print(f'dumping branch {branch_name}')     
            candidate_domain.to_cpu()
            #import pdb; pdb.set_trace()
            dump_branch(branch_name,
                        candidate_domain.mask,
                        candidate_domain.lower_bound,
                        candidate_domain.upper_bound,
                        candidate_domain.lower_all,
                        candidate_domain.upper_all,
                        candidate_domain.parent_solution,
                        candidate_domain.parent_ub_point,
                        gt_lb_relu,
                        final_branching_decision
                    )
        #import pdb; pdb.set_trace()
        nb_visited_states += 2
        # once the final branching decision is made, update and record
        # the resulted sub-domains

        # update the global upper bound (if necessary) comparing to the best of the batch
        #import pdb; pdb.set_trace()
        batch_ub, batch_ub_point_idx = torch.min(dom_ub, dim=0)
        batch_ub_point = dom_ub_point[batch_ub_point_idx]
        if batch_ub < global_ub:
            global_ub = batch_ub
            global_ub_point = batch_ub_point

        # sequentially add all the domains to the queue (ordered list)
        batch_global_lb = dom_lb[0]
        for batch_idx in range(dom_lb.shape[0]):
            print('dom_lb: ', dom_lb[batch_idx])
            print('dom_ub: ', dom_ub[batch_idx])

            if dom_lb[batch_idx] == float('inf') or dom_ub[batch_idx] == float('inf') or \
                    dom_lb[batch_idx] > dom_ub[batch_idx]:
                infeasible_count += 1

            elif dom_lb[batch_idx] < min(global_ub, decision_bound):
                c_dom_lb_all = [lb[batch_idx].unsqueeze(0) for lb in dom_lb_all]
                c_dom_ub_all = [ub[batch_idx].unsqueeze(0) for ub in dom_ub_all]
                c_updated_mask = [msk[batch_idx].unsqueeze(0) for msk in updated_mask]

                # If SaddleLP/Gurobi, the parent init is stored as a list of tensors for rho
                c_dual_solutions = [dsol[batch_idx].unsqueeze(0) for dsol in dual_solutions]

                dom_to_add = ReLUDomain(
                    c_updated_mask, lb=dom_lb[batch_idx].unsqueeze(0), ub=dom_ub[batch_idx].unsqueeze(0),
                    lb_all=c_dom_lb_all, up_all=c_dom_ub_all, parent_solution=c_dual_solutions,
                    parent_ub_point=dom_ub_point[batch_idx].unsqueeze(0)
                ).to_cpu()
                bab.add_domain(dom_to_add, domains)
                batch_global_lb = min(dom_lb[batch_idx], batch_global_lb)

        relu_end = time.time()
        print('A batch of relu splits requires: ', relu_end - relu_start)

        # Let's leave this commented out for the time being, it should either work without, or crash in all cases.
        # del dom_ub, dom_lb, dom_ub_all, dom_lb_all, updated_mask, dual_solutions
        # torch.cuda.empty_cache()

        # Update global LB.
        if len(domains) > 0:
            global_lb = domains[0].lower_bound.to(bounds_net_device)
        else:
            # If we've run out of domains, it means we included no newly splitted domain
            global_lb = torch.ones_like(global_lb) * (decision_bound + eps) if batch_global_lb > global_ub \
                else batch_global_lb

        # Remove domains clearly on the right side of the decision threshold: our goal is to which side of it is the
        # minimum, no need to know more for these domains.
        prune_value = min(global_ub.cpu() - eps, decision_bound + eps)
        domains = bab.prune_domains(domains, prune_value)

        print(f"Current: lb:{global_lb}\t ub: {global_ub}")

        # Stopping criterion
        if global_lb >= decision_bound:
            break
        elif global_ub < decision_bound:
            break

        if nb_visited_states > 1500:
            print('Early Termination for nb_visited_states>1500')
            return global_lb, global_ub, global_ub_point, nb_visited_states

        if not gt_throughout:
            print(f'Throughout OFF Mode: {branch_number}/{total_branches} collected.')
            if branch_number == total_branches:
                print('Finised collecting: Early termination')
                return global_lb, global_ub, global_ub_point, nb_visited_states


    bab.join_children(gurobi_dict, timeout)

    print(f"Terminated in {time.time() - start_time}[s]; {nb_visited_states} nodes.")
    print(f"Infeasible count: {infeasible_count}")

    return global_lb, global_ub, global_ub_point, nb_visited_states




def testing_indices(mask, score):
    '''
    select a representative subset of indices of the set of all available unfixed relu choices
    1. ensure at least 10% coverage 34+15+2
    2. include the top 40 kw choices (with preference giving to layer 1 and layer 2)
    =====> only need to augment the choices on layer 0
    '''
    selected_indices = {}
    for i in range(len(mask)):
        selected_indices[i] = []
    new_score = {}
    new_score_l2 = {}
    new_score_l1 = {}
    for i in range(len(score)):
        for j in range(len(score[i])):
            if mask[i][j].item()== -1:
                new_score[f'relu_{i}_{j}'] = score[i][j].item()
                if (i==1):
                    new_score_l1[f'relu_{i}_{j}'] = score[i][j].item()
                if (i==2):
                    new_score_l2[f'relu_{i}_{j}'] = score[i][j].item()
            else:
                pass

    
    new_score = sorted(new_score.items(), key = lambda x : x[1])
    new_score_l1 = sorted(new_score_l1.items(), key = lambda x : x[1])
    new_score_l2 = sorted(new_score_l2.items(), key = lambda x : x[1])
    new_score.reverse()
    new_score_l1.reverse()
    new_score_l2.reverse()
    kw_choices = new_score[:60]+new_score_l1[:20]+new_score_l2[:20]
    for i in set(kw_choices):
        selected_indices[int(i[0].split('_')[1])].append(int(i[0].split('_')[2]))

    for relu_idx in range(len(mask)-1, -1, -1):
        all_available_choices = (mask[relu_idx]==-1).nonzero().view(-1).tolist()
        required_number = int(len(all_available_choices)*0.1)
        done_choices = selected_indices[relu_idx]
        required_number = required_number - len(done_choices)
        ## DEBUG
        # if len(done_choices) == 0:
        if required_number <= 0:
            # No need to add points on this layer
            continue
        else:
            remained_choices = np.setdiff1d(all_available_choices, done_choices)
            selected_choices = np.random.choice(remained_choices, required_number, replace=False)
            selected_indices[relu_idx].extend(selected_choices)

    #print(selected_indices) 
    selected_branching_choices = []
    for key in selected_indices.keys():
        temp = [[key, item] for item in selected_indices[key]]
        selected_branching_choices.extend(temp)
    #import pdb; pdb.set_trace()
    return selected_branching_choices
