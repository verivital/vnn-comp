import torch
import copy

import plnn.branch_and_bound.utils as bab
from plnn.branch_and_bound.branching_scores import BranchingChoice
import time
from plnn.proxlp_solver import utils
from math import floor, ceil
from plnn.proxlp_solver.dj_relaxation import DJRelaxationLP


class ReLUDomain:
    '''
    Object representing a domain where the domain is specified by decision
    assigned to ReLUs.
    Comparison between instances is based on the values of
    the lower bound estimated for the instances.

    The domain is specified by `mask` which corresponds to a pattern of ReLUs.
    Neurons mapping to a  0 value are assumed to always have negative input (0 output slope)
          "               1                    "             positive input (1 output slope).
          "               -1 value are considered free and have no assumptions.

    For a MaxPooling unit, -1 indicates that we haven't picked a dominating input
    Otherwise, this indicates which one is the dominant one
    '''
    def __init__(self, mask, lb=-float('inf'), ub=float('inf'), lb_all=None, up_all=None, parent_solution=None,
                 parent_ub_point=None):
        self.mask = mask
        self.lower_bound = lb
        self.upper_bound = ub
        self.lower_all = lb_all
        self.upper_all = up_all
        self.parent_solution = parent_solution
        self.parent_ub_point = parent_ub_point

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

    def __le__(self, other):
        return self.lower_bound <= other.lower_bound

    def __eq__(self, other):
        return self.lower_bound == other.lower_bound

    def to_cpu(self):
        # transfer the content of this domain to cpu memory (try to reduce memory consumption)
        self.mask = [msk.cpu() for msk in self.mask]
        self.lower_bound = self.lower_bound.cpu()
        self.upper_bound = self.upper_bound.cpu()
        self.lower_all = [lbs.cpu() for lbs in self.lower_all]
        self.upper_all = [ubs.cpu() for ubs in self.upper_all]
        if self.parent_solution is not None:
            # TODO: when doing ExpLP, this needs to be coded more systematically (e.g., a ParentInit class)
            if not isinstance(self.parent_solution[0], tuple):
                # If SaddleLP/Gurobi, the parent init is stored as a list of tensors for rho
                self.parent_solution = [psol.cpu() for psol in self.parent_solution]
            else:
                # If DJRelaxationLP, the parent init is stored as a list of tuples of tensors for (lamba, mu)
                self.parent_solution = [(psollambda.cpu(), psolmu.cpu()) for psollambda, psolmu in self.parent_solution]
        if self.parent_ub_point is not None:
            self.parent_ub_point = self.parent_ub_point.cpu()
        return self

    def to_device(self, device):
        # transfer the content of this domain to cpu memory (try to reduce memory consumption)
        self.mask = [msk.to(device) for msk in self.mask]
        self.lower_bound = self.lower_bound.to(device)
        self.upper_bound = self.upper_bound.to(device)
        self.lower_all = [lbs.to(device) for lbs in self.lower_all]
        self.upper_all = [ubs.to(device) for ubs in self.upper_all]
        if self.parent_solution is not None:
            # TODO: when doing ExpLP, this needs to be coded more systematically (e.g., a ParentInit class)
            if not isinstance(self.parent_solution[0], tuple):
                # If SaddleLP/Gurobi, the parent init is stored as a list of tensors for rho
                self.parent_solution = [psol.to(device) for psol in self.parent_solution]
            else:
                # If DJRelaxationLP, the parent init is stored as a list of tuples of tensors for (lamba, mu)
                self.parent_solution = [(psollambda.to(device), psolmu.to(device)) for psollambda, psolmu in self.parent_solution]
        if self.parent_ub_point is not None:
            self.parent_ub_point = self.parent_ub_point.to(device)
        return self


def relu_bab(intermediate_net, bounds_net, branching_net_name, domain, decision_bound, eps=1e-4, sparsest_layer=0,
             timeout=float("inf"), batch_size=5, max_mem_consumption=100, parent_init_flag=True, 
             gurobi_specs=None, branching_threshold=0.2):
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
    `parent_init_flag`: whether to initialize every optimization from its parent node
    `gurobi_specs`: dictionary containing whether ("gurobi") gurobi needs to be used (executes on "p" cpu)
    Returns         : Lower bound and Upper bound on the global minimum,
                      as well as the point where the upper bound is achieved
    '''
    nb_visited_states = 0
    fail_safe_ratio = -1
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

    if intermediate_lbs[-1] > decision_bound or intermediate_ubs[-1] < decision_bound:
        bab.join_children(gurobi_dict, timeout)
        return intermediate_lbs[-1], intermediate_ubs[-1], \
               intermediate_net.get_lower_bound_network_input(), nb_visited_states, fail_safe_ratio


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
        return global_lb, global_ub, global_ub_point, nb_visited_states, fail_safe_ratio

    candidate_domain = ReLUDomain(updated_mask, lb=global_lb, ub=global_ub, lb_all=intermediate_lbs,
                                  up_all=intermediate_ubs, parent_solution=parent_init).to_cpu()

    domains = [candidate_domain]

    # initialise branching related terms
    branching_tools = BranchingChoice(updated_mask, sparsest_layer, intermediate_net.weights, branching_net_name)
    

    if branching_net_name is None:
        heuristic_choice = True
    else:
        heuristic_choice = False
        ## record parent_ub_point
        candidate_domain.parent_ub_point = global_ub_point.cpu()
        if isinstance(bounds_net, DJRelaxationLP):
            raise NotImplementedError
        domain_lower_bounds_stacks = []
        ## create more lists for the fail-safe strategy
        ## when gnn branching decision is not satisfactory,
        ## we move related domain info into *_previous lists 
        ## and call heuristic choices to see any improvement
        ## can be made.
        orig_ub_stacks_previous = []; orig_lb_stacks_previous = []
        orig_mask_stacks_previous = []
        parent_init_stacks_previous = None
        ## we also record already computed final bounds info,
        ## so if heuristic choices give worse performance, we
        ## can switch back to gnn decisions
        gnn_improvements_stacks_previous = []
        dom_ub_gnn=[]; dom_lb_gnn=[]; dom_ub_point_gnn=[] 
        updated_mask_gnn=[]; dom_lb_all_gnn=[]; dom_ub_all_gnn=[]
        dual_solutions_gnn =[]
        # performance evaluation
        heuristic_total = 0; gnn_total = 0
    
    infeasible_count = 0
    while global_ub - global_lb > eps:

        # Check if we have run out of time.
        if time.time() - start_time > timeout:
            bab.join_children(gurobi_dict, timeout)
            return None, None, None, nb_visited_states, fail_safe_ratio

        ## since branching decisions are processed in batches, we collect 
        ## all necessary domain info with the following lists

        orig_ub_stacks_current = []
        orig_lb_stacks_current = []
        orig_mask_stacks_current = [] 
        if not heuristic_choice:
            orig_gl_lb_stacks_current = []
            orig_parent_ub_points_stacks_current=[]
            orig_parent_sol_stacks_current=[]
            effective_batch_size = min(batch_size, len(domains)+len(orig_ub_stacks_previous))
        else:
            effective_batch_size = min(batch_size, len(domains))
        print(f"effective_batch_size {effective_batch_size}")

        # effective_batch_size*2 as every candidate domain is split in two different ways
        splitted_lbs_stacks = [lbs[0].to(intermediate_net_device).unsqueeze(0).repeat(((effective_batch_size*2,) + (1,) * (lbs.dim() - 1)))
                               for lbs in bounds_net.lower_bounds]
        splitted_ubs_stacks = [ubs[0].to(intermediate_net_device).unsqueeze(0).repeat(((effective_batch_size*2,) + (1,) * (ubs.dim() - 1)))
                               for ubs in bounds_net.upper_bounds]
        splitted_domain = domain.unsqueeze(0).expand(((effective_batch_size*2,) + (-1,) * domain.dim()))

        if not heuristic_choice:
            effective_batch_size = effective_batch_size - len(orig_lb_stacks_previous)
            print(f'effective_batch_size for new domains: {effective_batch_size}')
        # TODO: This needs to be coded more systematically (e.g., a ParentInit class)
        if not isinstance(bounds_net, DJRelaxationLP):
            # If SaddleLP/Gurobi, the parent init is stored as a list of tensors for rho
            parent_init_stacks = [pinits[0].unsqueeze(0).repeat(((effective_batch_size*2,) + (1,) * (pinits.dim() - 1)))
                                  for pinits in bounds_net.last_duals]
        else:
            # If DJRelaxationLP, the parent init is stored as a list of tuples of tensors for (lamba, mu)
            parent_init_stacks = [
                (pinits_lambda[0].unsqueeze(0).repeat(((effective_batch_size * 2,) + (1,) * (pinits_lambda.dim() - 1))),
                 pinits_mu[0].unsqueeze(0).repeat(((effective_batch_size * 2,) + (1,) * (pinits_mu.dim() - 1))),)
                for pinits_lambda, pinits_mu in bounds_net.last_duals]

        # List of sets storing for each layer index, the batch entries that are splitting a ReLU there
        branching_layer_log = []
        for _ in range(len(intermediate_net.lower_bounds)-1):
            branching_layer_log.append(set())
       
        for batch_idx in range(effective_batch_size):
            # Pick a domain to branch over and remove that from our current list of
            # domains. Also, potentially perform some pruning on the way.
            candidate_domain = bab.pick_out(domains, global_ub.cpu() - eps).to_device(intermediate_net_device)
            # Generate new, smaller domains by splitting over a ReLU
            mask = candidate_domain.mask
            orig_lbs = candidate_domain.lower_all
            orig_ubs = candidate_domain.upper_all

            # collect branching related information
            orig_lb_stacks_current.append(orig_lbs)
            orig_ub_stacks_current.append(orig_ubs)
            orig_mask_stacks_current.append(mask)
            
            # collect more information if gnn branching decision is used
            if not heuristic_choice:
                orig_gl_lb_stacks_current.append(candidate_domain.lower_bound.squeeze(0))
                domain_lower_bounds_stacks.append(candidate_domain.lower_bound)
                if candidate_domain.parent_solution[0].size()[1]==2:
                    duals_reshape = [i.squeeze(0).view(2,-1).T for i in candidate_domain.parent_solution]
                    orig_parent_sol_stacks_current.append(duals_reshape)
                ## Dual Var reshape has only been implemented for prox method for now
                ## the exception is put here in case other methods output different dual forms
                else:
                    raise NotImplementedError
                orig_parent_ub_points_stacks_current.append(candidate_domain.parent_ub_point)
               
            # get parent's dual solution from the candidate domain
            for x_idx in range(len(parent_init_stacks)):
                # TODO: This needs to be coded more systematically (e.g., a ParentInit class)
                if not isinstance(bounds_net, DJRelaxationLP):
                    # If SaddleLP/Gurobi, the parent init is stored as a list of tensors for rho
                    parent_init_stacks[x_idx][2*batch_idx] = candidate_domain.parent_solution[x_idx].clone()
                    parent_init_stacks[x_idx][2*batch_idx + 1] = candidate_domain.parent_solution[x_idx].clone()
                else:
                    # If DJRelaxationLP, the parent init is stored as a list of tuples of tensors for (lamba, mu)
                    parent_init_stacks[x_idx][0][2 * batch_idx] = candidate_domain.parent_solution[x_idx][0].clone()
                    parent_init_stacks[x_idx][1][2 * batch_idx] = candidate_domain.parent_solution[x_idx][1].clone()
                    parent_init_stacks[x_idx][0][2 * batch_idx + 1] = candidate_domain.parent_solution[x_idx][0].clone()
                    parent_init_stacks[x_idx][1][2 * batch_idx + 1] = candidate_domain.parent_solution[x_idx][1].clone()

        # Compute branching choices
        # TODO: branching will return IndexError in case no ambiguous ReLU is left. Should catch this, and compute
        if heuristic_choice:
            branching_decision_list, _ = branching_tools.heuristic_branching_decision(orig_lb_stacks_current, orig_ub_stacks_current, orig_mask_stacks_current)

        else:
            branching_decision_list = []
            if len(orig_lb_stacks_previous)!= 0:
                branching_decision_list_temp, _ = branching_tools.heuristic_branching_decision(orig_lb_stacks_previous, orig_ub_stacks_previous, orig_mask_stacks_previous)
                branching_decision_list += branching_decision_list_temp
                #print('heuristic: ',branching_decision_list_temp)
            if effective_batch_size > 0:
                branching_decision_list += branching_tools.gnn_branching_decision(orig_lb_stacks_current, orig_ub_stacks_current, orig_parent_sol_stacks_current, orig_parent_ub_points_stacks_current, orig_gl_lb_stacks_current, orig_mask_stacks_current)
            #print('overall: ',branching_decision_list)

                
        if not heuristic_choice:
            orig_lb_stacks_current = orig_lb_stacks_previous + orig_lb_stacks_current 
            orig_ub_stacks_current = orig_ub_stacks_previous + orig_ub_stacks_current 
            if parent_init_stacks_previous is not None: 
                parent_init_stacks_temp = [torch.cat([i,j],0) for i,j in zip(parent_init_stacks_previous, parent_init_stacks)]
                parent_init_stacks = parent_init_stacks_temp
        
        for batch_idx, branching_decision in enumerate(branching_decision_list):
            branching_layer_log[branching_decision[0]] |= {2*batch_idx, 2*batch_idx+1}
            orig_lbs  =  orig_lb_stacks_current[batch_idx]
            orig_ubs  =  orig_ub_stacks_current[batch_idx]

            for choice in [0, 1]:
                print(f'splitting decision: {branching_decision} - choice {choice}')
                # Find the upper and lower bounds on the minimum in the domain
                # defined by n_mask_i
                nb_visited_states += 1
                if (nb_visited_states % 10) == 0:
                    print(f"Running Nb states visited: {nb_visited_states}")
                    print(f"N. infeasible nodes {infeasible_count}")

                # split the domain with the current branching decision
                splitted_lbs_stacks, splitted_ubs_stacks = update_bounds_from_split(
                    branching_decision, choice, orig_lbs, orig_ubs, 2*batch_idx + choice, splitted_lbs_stacks,
                    splitted_ubs_stacks)
                    

        relu_start = time.time()
        # compute the bounds on the batch of splits, at once
        dom_ub_temp, dom_lb_temp, dom_ub_point_temp, updated_mask_temp, dom_lb_all_temp, dom_ub_all_temp, dual_solutions_temp = compute_bounds(
            intermediate_net, bounds_net, branching_layer_log, splitted_domain, splitted_lbs_stacks, splitted_ubs_stacks,
            parent_init_stacks, parent_init_flag, gurobi_dict
        )
        # SANITY for bounds problems
        #for idx in range(len(dom_lb_all_temp)):
        #    for sub_idx in range(len(dom_lb_all_temp[idx])):
        #        if torch.sum(dom_ub_all_temp[idx][sub_idx]<dom_lb_all_temp[idx][sub_idx])!=0:
        #            if dom_lb_temp[sub_idx] < 0:
        #                fail_safe_ratio = -100
        #                #return global_lb, global_ub, global_ub_point, nb_visited_states, fail_safe_ratio
        #                import pdb; pdb.set_trace()

        # fail-safe for branching choices
        if not heuristic_choice:
            dom_ub = []; dom_lb = []; dom_ub_point = []; updated_mask=[]
            dom_lb_all = []; dom_ub_all = []; dual_solutions = []
            results_final = [dom_ub, dom_lb, dom_ub_point] 
            results_final_lists = [updated_mask,dom_lb_all, dom_ub_all, dual_solutions]
            results_gnn = [dom_ub_gnn, dom_lb_gnn, dom_ub_point_gnn]
            results_gnn_lists = [updated_mask_gnn, dom_lb_all_gnn, dom_ub_all_gnn, dual_solutions_gnn]
            results_temp = [dom_ub_temp, dom_lb_temp, dom_ub_point_temp] 
            results_temp_lists = [updated_mask_temp,dom_lb_all_temp, dom_ub_all_temp, dual_solutions_temp]
            previous_chunk_size = len(orig_lb_stacks_previous)
            # the output results consists of two chunks. The first chunk contains
            for batch_idx_prev in range(previous_chunk_size):
                heuristic_improvement = ((min(dom_lb_temp[2*batch_idx_prev],0) + min(dom_lb_temp[2*batch_idx_prev+1],0)) - 2*domain_lower_bounds_stacks[batch_idx_prev][0])/(-2*domain_lower_bounds_stacks[batch_idx_prev][0])
                gnn_improvement = gnn_improvements_stacks_previous[batch_idx_prev]
                if gnn_improvement >= heuristic_improvement:
                    # use previous gnn results
                    gnn_total += 1
                    bab.add_terms(results_final, results_gnn, batch_idx_prev)
                    bab.add_terms_lists(results_final_lists, results_gnn_lists, batch_idx_prev)
                else:
                    # use new results computed with heuristic choices
                    heuristic_total += 1
                    bab.add_terms(results_final, results_temp, batch_idx_prev)
                    bab.add_terms_lists(results_final_lists, results_temp_lists, batch_idx_prev)

            print(f'Resolved undecided domains: {previous_chunk_size}')
            domain_lower_bounds_stacks_temp = []
            # clear previous recorded information
            orig_ub_stacks_previous=[]
            orig_lb_stacks_previous=[] 
            orig_mask_stacks_previous=[]
            gnn_improvements_stacks_previous=[]
            parent_init_stacks_previous = None
            for item in results_gnn: item[:] = []
            for item in results_gnn_lists: item[:] = []

            heuristic_required_counter = 0
            # re-compute results with heuristic branching choices
            recompute_indices = []
            #import pdb; pdb.set_trace()
            effective_batch_size = len(orig_ub_stacks_current)
            for batch_idx in range(previous_chunk_size, effective_batch_size):
                gnn_improvement = ((min(dom_lb_temp[2*batch_idx],0) + min(dom_lb_temp[2*batch_idx+1],0)) - 2*domain_lower_bounds_stacks[batch_idx][0])/(-2*domain_lower_bounds_stacks[batch_idx][0])
                print(gnn_improvement)
                if gnn_improvement < branching_threshold:
                    # gnn branching decision is not satisfactory
                    # record all information and try again with heuristic choices
                    heuristic_required_counter += 1
                    recompute_indices.append(batch_idx-previous_chunk_size)
                    # first record all improvement info
                    gnn_improvements_stacks_previous.append(gnn_improvement)
                    domain_lower_bounds_stacks_temp.append(domain_lower_bounds_stacks[batch_idx])

                    # then record all info for making heuristic branching decisions
                    orig_ub_stacks_previous.append(orig_ub_stacks_current[batch_idx])
                    orig_lb_stacks_previous.append(orig_lb_stacks_current[batch_idx])
                    orig_mask_stacks_previous.append(orig_mask_stacks_current[batch_idx-previous_chunk_size])
                    # finally record all computed bounds info in case we need to
                    # switch back to gnn decisions when heuristic choices are worse
                    bab.add_terms(results_gnn, results_temp, batch_idx)
                    bab.add_terms_lists(results_gnn_lists, results_temp_lists, batch_idx)
                else:
                    # gnn performance is satisfactory 
                    gnn_total += 1
                    bab.add_terms(results_final, results_temp, batch_idx)
                    bab.add_terms_lists(results_final_lists, results_temp_lists, batch_idx)

            if len(recompute_indices)!=0:
                # the following could be simplified if it is confirmed parent_init_stacks
                # will not be modified by compute_bounds
                recompute_size = len(recompute_indices)
                parent_init_stacks_previous = [pinits[0].unsqueeze(0).repeat(((recompute_size*2,) + (1,) * (pinits.dim() - 1)))
                                          for pinits in bounds_net.last_duals]
                for x_idx in range(len(parent_init_stacks_previous)):
                    shape = candidate_domain.parent_solution[x_idx].size()
                    for idx, re_idx in enumerate(recompute_indices):
                        parent_init_stacks_previous[x_idx][2*idx] = orig_parent_sol_stacks_current[re_idx][x_idx].T.unsqueeze(0).reshape(shape).clone()
                        parent_init_stacks_previous[x_idx][2*idx+1] = orig_parent_sol_stacks_current[re_idx][x_idx].T.unsqueeze(0).reshape(shape).clone()

            domain_lower_bounds_stacks = domain_lower_bounds_stacks_temp
            #import pdb; pdb.set_trace()
            print(f'Number of domains that require heuristic recomputation: {heuristic_required_counter}')
            print(f'gnn_total: {gnn_total}')
            print(f'heu_total: {heuristic_total}')

        if not heuristic_choice:
            if len(dom_ub) == 0:
                continue
            else:
                dom_ub, dom_lb, dom_ub_point, updated_mask,dom_lb_all, dom_ub_all, dual_solutions = bab.modify_forms(results_final, results_final_lists)
                #import pdb; pdb.set_trace()
        else:
            dom_ub=dom_ub_temp; dom_lb=dom_lb_temp; dom_ub_point=dom_ub_point_temp
            updated_mask= updated_mask_temp; dom_lb_all= dom_lb_all_temp
            dom_ub_all= dom_ub_all_temp; dual_solutions= dual_solutions_temp

        # update the global upper bound (if necessary) comparing to the best of the batch
        batch_ub, batch_ub_point_idx = torch.min(dom_ub, dim=0)
        batch_ub_point = dom_ub_point[batch_ub_point_idx]
        if batch_ub < global_ub:
            global_ub = batch_ub
            global_ub_point = batch_ub_point

        for batch_idx in range(updated_mask[0].shape[0]):
            current_tot_ambi_nodes = 0
            for layer_mask in updated_mask:
                current_tot_ambi_nodes += torch.sum(layer_mask[batch_idx] == -1).item()
            # print(f"total number of ambiguous nodes: {current_tot_ambi_nodes}")

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

                # TODO: when doing ExpLP, this needs to be coded more systematically (e.g., a ParentInit class)
                if not isinstance(bounds_net, DJRelaxationLP):
                    # If SaddleLP/Gurobi, the parent init is stored as a list of tensors for rho
                    c_dual_solutions = [dsol[batch_idx].unsqueeze(0) for dsol in dual_solutions]
                else:
                    # If DJRelaxationLP, the parent init is stored as a list of tuples of tensors for (lamba, mu)
                    c_dual_solutions = [(lambdasol[batch_idx].unsqueeze(0), musol[batch_idx].unsqueeze(0))
                                        for lambdasol, musol in dual_solutions]

                dom_to_add = ReLUDomain(
                    c_updated_mask, lb=dom_lb[batch_idx].unsqueeze(0), ub=dom_ub[batch_idx].unsqueeze(0),
                    lb_all=c_dom_lb_all, up_all=c_dom_ub_all, parent_solution=c_dual_solutions
                ).to_cpu()
                if not heuristic_choice:
                    dom_to_add.parent_ub_point = dom_ub_point[batch_idx].cpu()
                bab.add_domain(dom_to_add, domains)
                batch_global_lb = min(dom_lb[batch_idx], batch_global_lb)

        relu_end = time.time()
        print('A batch of relu splits requires: ', relu_end - relu_start)

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

    bab.join_children(gurobi_dict, timeout)

    print(f"Terminated in {time.time() - start_time}[s]; {nb_visited_states} nodes.")
    print(f"Infeasible count: {infeasible_count}")

    if not heuristic_choice:
        fail_safe_ratio = heuristic_total/(gnn_total+heuristic_total)

    return global_lb, global_ub, global_ub_point, nb_visited_states, fail_safe_ratio


def update_bounds_from_split(decision, choice, old_lbs, old_ubs, batch_idx, splitted_lbs_stacks, splitted_ubs_stacks):
    """
    Given a ReLU branching decision and bounds for all the activations, clip the bounds according to the decision.
    Update performed in place in the list of lower/upper bound stacks (batches of lower/upper bounds)
    :param decision: tuples (x_idx, node)
    :param choice: 0/1 for whether to clip on a blocking/passing ReLU
    :param old_lbs: list of tensors for the (pre-activation) lower bounds relative to all the activations of the network
    :param old_ubs:list of tensors for the (pre-activation) upper bounds relative to all the activations of the network
    :param splitted_lbs_stacks: batched lower bounds to update with the splitted ones at batch_idx
    :param splitted_ubs_stacks: batched upper bounds to update with the splitted ones at batch_idx
    """
    new_ubs = copy.deepcopy(old_ubs)
    new_lbs = copy.deepcopy(old_lbs)

    assert new_lbs[0].shape[0] == 1

    if decision is not None:
        change_idx = decision[0] + 1
        # upper_bound for the corresponding relu is forced to be 0
        if choice == 0:
            # blocking ReLU obtained by setting the pre-activation UB to 0
            new_ubs[change_idx].view(-1)[decision[1]] = 0
        else:
            # passing ReLU obtained by setting the pre-activation LB to 0
            new_lbs[change_idx].view(-1)[decision[1]] = 0

    for x_idx in range(len(splitted_lbs_stacks)):
        splitted_lbs_stacks[x_idx][batch_idx] = new_lbs[x_idx]
        splitted_ubs_stacks[x_idx][batch_idx] = new_ubs[x_idx]
    return splitted_lbs_stacks, splitted_ubs_stacks


def compute_bounds(intermediate_net, bounds_net, branching_layer_log, splitted_domain, splitted_lbs,
                   splitted_ubs, parent_init_stacks, parent_init_flag, gurobi_dict):
    """
    Split domain according to branching decision and compute all the necessary quantities for it.
    Splitting on the input domain will never happen as it'd be done on l1-u1, rather than l0-u0 (representing the
    conditioned input domain). So conditioning is not problematic, here.
    :param intermediate_net: Network used for intermediate bounds
    :param bounds_net: Network used for last bounds
    :param branching_layer_log: List of sets storing for each layer index, the set of batch entries that are
        splitting a ReLU there (stored like x_idx-1)
    :param choice: 0/1 for whether to clip on a blocking/passing ReLU
    :param splitted_lbs: list of tensors for the (pre-activation) lower bounds relative to all the activations of the
    network, for all the domain batches
    :param splitted_ubs:list of tensors for the (pre-activation) upper bounds relative to all the activations of the
        network, for all the domain batches
    :param parent_init_stacks:list of tensors to use as dual variable initialization in the last layer solver
    :return: domain UB, domain LB, net input point that yielded UB, updated ReLU mask, updated old_lbs, updated old_ubs
    :param parent_init_flag: whether to initialize the bounds optimisation from the parent node
    :param gurobi_dict: dictionary containing information for gurobi's (possibly parallel) execution
    """
    # update intermediate bounds after the splitting
    splitted_lbs, splitted_ubs = compute_intermediate_bounds(
        intermediate_net, branching_layer_log, splitted_domain, splitted_lbs, splitted_ubs)

    # update and retrieve which relus are active/passing/ambiguous (need to rebuild the model with all the batch)
    intermediate_net.build_model_using_bounds(splitted_domain, (splitted_lbs, splitted_ubs))
    intermediate_net.update_relu_mask()
    updated_mask = intermediate_net.relu_mask

    # get the new last-layer bounds after the splitting
    if not gurobi_dict["gurobi"]:
        # compute all last layer bounds in parallel
        if parent_init_flag:
            bounds_net.initialize_dual_from(parent_init_stacks)

        bounds_net.build_model_using_bounds(splitted_domain, (splitted_lbs, splitted_ubs))
        updated_lbs, updated_ubs = bounds_net.compute_lower_bound(counterexample_verification=True)
        # here, I could do the lower bound only to save memory, but it's useful to check for infeasibility
        splitted_lbs[-1] = torch.max(updated_lbs, splitted_lbs[-1])
        splitted_ubs[-1] = torch.min(updated_ubs, splitted_ubs[-1])
        # evaluate the network at the lower bound point
        dom_ub_point = bounds_net.get_lower_bound_network_input()
        dual_solutions = bounds_net.last_duals
    else:
        # compute them one by one
        splitted_lbs, splitted_ubs, dom_ub_point, dual_solutions = compute_last_bounds_cpu(
            bounds_net, splitted_domain, splitted_lbs, splitted_ubs, gurobi_dict)

    # retrieve bounds info from the bounds network: the lower bounds are the output of the bound calculation, the upper
    # bounds are computed by evaluating the network at the lower bound points.
    dom_lb_all = splitted_lbs
    dom_ub_all = splitted_ubs
    dom_lb = splitted_lbs[-1]


    # TODO: do we need any alternative upper bounding strategy for the dual algorithms?
    dom_ub = bounds_net.net(dom_ub_point)

    # check that the domain upper bound is larger than its lower bound. If not, infeasible domain (and mask).
    # return +inf as a consequence to have the bound pruned.
    primal_feasibility = bab.check_primal_infeasibility(dom_lb_all, dom_ub_all, dom_lb, dom_ub)
    dom_lb = torch.where(~primal_feasibility, float('inf') * torch.ones_like(dom_lb), dom_lb)
    dom_ub = torch.where(~primal_feasibility, float('inf') * torch.ones_like(dom_ub), dom_ub)

    return dom_ub, dom_lb, dom_ub_point, updated_mask, dom_lb_all, dom_ub_all, dual_solutions


def compute_intermediate_bounds(intermediate_net, branching_layer_log, splitted_domain, intermediate_lbs,
                                intermediate_ubs):
    # compute intermediate bounds for the current batch, leaving out unnecessary computations
    # (those before the splitted relus)

    # get minimum layer idx where branching is happening
    min_branching_layer = len(intermediate_net.weights)
    for branch_lay_idx in range(len(branching_layer_log)):
        if branching_layer_log[branch_lay_idx]:
            min_branching_layer = branch_lay_idx
            break

    # List of sets storing for each layer index, the batch entries that are splitting a ReLU there or onwards
    cumulative_branching_layer_log = [None] * (len(intermediate_net.lower_bounds)-1)
    for branch_lay_idx in range(len(branching_layer_log)):
        cumulative_branching_layer_log[branch_lay_idx] = branching_layer_log[branch_lay_idx]
        if branch_lay_idx > 0:
            cumulative_branching_layer_log[branch_lay_idx] |= cumulative_branching_layer_log[branch_lay_idx-1]

    # TODO: this was +1 in the PLNN-verification-private codebase, but this should be correct (and is more efficient)
    for x_idx in range(min_branching_layer+2, len(intermediate_net.lower_bounds)):

        active_batch_ids = list(cumulative_branching_layer_log[x_idx-2])
        sub_batch_intermediate_lbs = [lbs[active_batch_ids] for lbs in intermediate_lbs]
        sub_batch_intermediate_ubs = [ubs[active_batch_ids] for ubs in intermediate_ubs]

        intermediate_net.build_model_using_bounds(
            splitted_domain[active_batch_ids],
            (sub_batch_intermediate_lbs, sub_batch_intermediate_ubs))
        updated_lbs, updated_ubs = intermediate_net.compute_lower_bound(
            node=(x_idx, None), counterexample_verification=True)

        # retain best bounds and update intermediate bounds from batch
        intermediate_lbs[x_idx][active_batch_ids] = torch.max(updated_lbs, intermediate_lbs[x_idx][active_batch_ids])
        intermediate_ubs[x_idx][active_batch_ids] = torch.min(updated_ubs, intermediate_ubs[x_idx][active_batch_ids])


    return intermediate_lbs, intermediate_ubs


def compute_last_bounds_cpu(bounds_net, splitted_domain, splitted_lbs, splitted_ubs, gurobi_dict):
    # Compute the last layer bounds on (multiple, if p>1) cpu over the batch domains (used for Gurobi).

    # Retrieve execution specs.
    p = gurobi_dict["p"]
    server_queue = gurobi_dict["server_queue"]
    instruction_queue = gurobi_dict["instruction_queue"]
    barrier = gurobi_dict["barrier"]

    if p == 1:
        batch_indices = list(range(splitted_lbs[0].shape[0]))
        cpu_splitted_domain, cpu_splitted_lbs, cpu_splitted_ubs = bab.subproblems_to_cpu(
            splitted_domain, splitted_lbs, splitted_ubs)
        splitted_lbs, splitted_ubs, dom_ub_point, dual_solutions = bab.compute_last_bounds_sequentially(
            bounds_net, cpu_splitted_domain, cpu_splitted_lbs, cpu_splitted_ubs, batch_indices)
    else:
        # Full synchronization after every batch.
        barrier.wait()

        cpu_splitted_domain, cpu_splitted_lbs, cpu_splitted_ubs = bab.subproblems_to_cpu(
            splitted_domain, splitted_lbs, splitted_ubs, share=True)

        max_batch_size = cpu_splitted_lbs[0].shape[0]
        c_batch_size = int(ceil(max_batch_size / float(p)))
        busy_processors = int(ceil(max_batch_size / float(c_batch_size))) - 1
        idle_processors = p - (busy_processors+1)

        # Send bounding jobs to the busy cpu servers.
        for sub_batch_idx in range(busy_processors):
            start_batch_index = sub_batch_idx * c_batch_size
            end_batch_index = min((sub_batch_idx + 1) * c_batch_size, max_batch_size)
            slice_indices = list(range(start_batch_index, end_batch_index))
            instruction_queue.put((cpu_splitted_domain, cpu_splitted_lbs, cpu_splitted_ubs, slice_indices))
        # Keep the others idle.
        for _ in range(idle_processors):
            instruction_queue.put(("idle",))

        # Execute the last sub-batch of bounds on this cpu core.
        slice_indices = list(range((busy_processors) * c_batch_size, max_batch_size))
        splitted_lbs, splitted_ubs, c_dom_ub_point, c_dual_solutions = bab.compute_last_bounds_sequentially(
            bounds_net, cpu_splitted_domain, cpu_splitted_lbs, cpu_splitted_ubs, slice_indices, share=True)

        # Gather by-products of bounding in the same format returned by a gpu-batched bounds computation.
        dom_ub_point = c_dom_ub_point[0].unsqueeze(0).repeat(((max_batch_size,) + (1,) * (c_dom_ub_point.dim() - 1)))
        dual_solutions = [cdsol[0].unsqueeze(0).repeat(((max_batch_size,) + (1,) * (cdsol.dim() - 1)))
                          for cdsol in c_dual_solutions]
        dom_ub_point[slice_indices] = c_dom_ub_point
        for idx in range(len(dual_solutions)):
            dual_solutions[idx][slice_indices] = c_dual_solutions[idx]

        for _ in range(busy_processors):
            # Collect bounding jobs from cpu servers.
            splitted_lbs, splitted_ubs, c_dom_ub_point, c_dual_solutions, slice_indices = \
                server_queue.get(True)

            # Gather by-products of bounding in the same format returned by a gpu-batched bounds computation.
            dom_ub_point[slice_indices] = c_dom_ub_point
            for idx in range(len(dual_solutions)):
                dual_solutions[idx][slice_indices] = c_dual_solutions[idx]

    return splitted_lbs, splitted_ubs, dom_ub_point, dual_solutions
