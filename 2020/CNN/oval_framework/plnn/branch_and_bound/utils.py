import bisect
import torch
import torch.multiprocessing as mp
import copy
import traceback


class CandidateDomain:
    '''
    Object representing a domain as produced by the BranchAndBound algorithm.
    Comparison between its elements is based on the values of the lower bounds
    that are estimated for it.
    '''
    def __init__(self, lb=-float('inf'), ub=float('inf'), dm=None):
        self.lower_bound = lb
        self.upper_bound = ub
        self.domain = dm

    def __lt__(self, other):
        return self.lower_bound < other.lower_bound

    def __le__(self, other):
        return self.lower_bound <= other.lower_bound

    def __eq__(self, other):
        return self.lower_bound == other.lower_bound

    def __repr__(self):
        string = f"[LB: {self.lower_bound:.4e}\t" \
                 f" UB:  {self.upper_bound:.4e}\n" \
                 f" Domain: {self.domain}]"
        return string

    def area(self):
        '''
        Compute the area of the domain
        '''
        dom_sides = self.domain.select(1, 1) - self.domain.select(1, 0)
        dom_area = dom_sides.prod()
        return dom_area


def add_domain(candidate, domains):
    '''
    Use binary search to add the new domain `candidate`
    to the candidate list `domains` so that `domains` remains a sorted list.
    '''
    bisect.insort_left(domains, candidate)


def pick_out(domains, threshold):
    '''
    Pick the first domain in the `domains` sequence
    that has a lower bound lower than `threshold`.

    Any domain appearing before the chosen one but having a lower_bound greater
    than the threshold is discarded.

    Returns: Non prunable CandidateDomain with the lowest reference_value.
    '''
    assert len(domains) > 0, "The given domains list is empty."
    while True:
        assert len(domains) > 0, "No domain left to pick from."
        selected_candidate_domain = domains.pop(0)
        if selected_candidate_domain.lower_bound < threshold:
            break

    return selected_candidate_domain


def n_satisfying_threshold(domains, threshold):
    '''
    Count all the domains in the `domains` sequence
    that have a lower bound lower than `threshold`.

    Returns: int, number of constraints satisfying threshold condition
    '''
    count = 0
    for candidate_domain in domains:
        if candidate_domain.lower_bound < threshold:
            count += 1
    return count


def box_split(domain):
    '''
    Use box-constraints to split the input domain.
    Split by dividing the domain into two from its longest edge.
    Assumes a rectangular domain, which is aligned with the cartesian
    coordinate frame.

    `domain`:  A 2d tensor whose rows contain lower and upper limits
               of the corresponding dimension.
    Returns: A list of sub-domains represented as 2d tensors.
    '''
    # Find the longest edge by checking the difference of lower and upper
    # limits in each dimension.
    diff = domain[:, 1] - domain[:, 0]
    edgelength, dim = torch.max(diff, 0)

    # Unwrap from tensor containers
    edgelength = edgelength.item()
    dim = dim.item()

    # Now split over dimension dim:
    half_length = edgelength/2

    # dom1: Upper bound in the 'dim'th dimension is now at halfway point.
    dom1 = domain.clone()
    dom1[dim, 1] -= half_length

    # dom2: Lower bound in 'dim'th dimension is now at haflway point.
    dom2 = domain.clone()
    dom2[dim, 0] += half_length

    sub_domains = [dom1, dom2]

    return sub_domains

def smart_box_split(ndomain, dualnet, domain_lb, domain_width, useful_cutoff):
    '''
    Use box-constraints to split the input domain.
    Split by dividing the domain into two.
    We decide on which dimension to split by trying all splits with a cheap lower bound.

    `domain`:  A 2d tensor whose rows contain lower and upper limits
               of the corresponding dimension.
    Returns: A list of sub-domains represented as 2d tensors.
    '''
    # We're going to try all possible combinations and get the bounds for each,
    # and pick the one with the largest (lowest lower bound of the two part)
    domain = domain_lb + domain_width * ndomain
    largest_lowest_lb = -float('inf')
    largest_lowest_lb_dim = None
    split_lbs = None
    for dim in range(domain.shape[0]):
        # Split alongst the i-th dimension

        dom1 = domain.clone()
        dom1[dim, 1] = (dom1[dim, 1] + dom1[dim, 0]) / 2
        dom2 = domain.clone()
        dom2[dim, 0] = (dom2[dim, 1] + dom2[dim, 0]) / 2

        both_doms = torch.stack([dom1, dom2], 0)

        lbs = dualnet.get_lower_bounds(both_doms)

        lowest_lb = lbs.min()
        if lowest_lb > largest_lowest_lb:
            largest_lowest_lb = lowest_lb
            largest_lowest_lb_dim = dim
            split_lbs = lbs

    ndom1 = ndomain.clone()
    ndom1[largest_lowest_lb_dim, 1] = (ndom1[largest_lowest_lb_dim, 1] + ndom1[largest_lowest_lb_dim, 0]) / 2
    ndom2 = ndomain.clone()
    ndom2[largest_lowest_lb_dim, 0] = (ndom2[largest_lowest_lb_dim, 1] + ndom2[largest_lowest_lb_dim, 0]) / 2

    sub_domains = [ndom1, ndom2]

    return sub_domains


def prune_domains(domains, threshold):
    '''
    Remove domain from `domains`
    that have a lower_bound greater than `threshold`
    '''
    # TODO: Could do this with binary search rather than iterating.
    # TODO: If this is not sorted according to lower bounds, this
    # implementation is incorrect because we can not reason about the lower
    # bounds of the domain that come after
    for i in range(len(domains)):
        if domains[i].lower_bound >= threshold:
            domains = domains[0:i]
            break
    return domains


def print_remaining_domain(domains):
    '''
    Iterate over all the domains, measuring the part of the whole input space
    that they contain and print the total share it represents.
    '''
    remaining_area = 0
    for dom in domains:
        remaining_area += dom.area()
    print(f'Remaining portion of the input space: {remaining_area*100:.8f}%')


def compute_last_bounds_sequentially(bounds_net, splitted_domain, splitted_lbs, splitted_ubs, batch_indices, share=False):
    # Compute the last layer bounds sequentially over the batch domains (used for Gurobi).

    for batch_idx in batch_indices:
        bounds_net.build_model_using_bounds(
            splitted_domain[batch_idx],
            ([lbs[batch_idx] for lbs in splitted_lbs],
             [ubs[batch_idx] for ubs in splitted_ubs]))
        updated_lbs = bounds_net.compute_lower_bound(node=(-1, 0), counterexample_verification=True)
        splitted_lbs[-1][batch_idx] = torch.max(updated_lbs, splitted_lbs[-1][batch_idx])
        # Store the output of the bounding procedure in a format consistent with the batched version.

        if batch_idx == batch_indices[0]:
            dom_ub_point = bounds_net.get_lower_bound_network_input().clone()
            dual_solutions = [c_duals.clone() for c_duals in bounds_net.last_duals]
        else:
            dom_ub_point = torch.cat([dom_ub_point, bounds_net.get_lower_bound_network_input()], 0)
            dual_solutions = [torch.cat([dual_solutions[idx], bounds_net.last_duals[idx]], 0)
                              for idx in range(len(dual_solutions))]
        if share:
            dom_ub_point = share_tensors(dom_ub_point)
            dual_solutions = share_tensors(dual_solutions)

    return splitted_lbs, splitted_ubs, dom_ub_point, dual_solutions


def check_primal_infeasibility(dom_lb_all, dom_ub_all, dom_lb, dom_ub):
    """
    Given intermediate bounds (lists of tensors) and final layer bounds, check whether these constitute an infeasible
    primal problem.
    This is checked via the dual, which is unbounded (lbs are larger than ubs, as we don't go to convergence).
    """
    batch_shape = dom_lb_all[0].shape[:1]
    feasible_output = torch.ones((*batch_shape, 1), device=dom_lb_all[0].device, dtype=torch.bool)
    for c_lbs, c_ubs in zip(dom_lb_all, dom_ub_all):
        feasible_output = feasible_output & (c_ubs - c_lbs >= 0).view((*batch_shape, -1)).all(dim=-1, keepdim=True)
    feasible_output = feasible_output & (dom_ub - dom_lb >= 0).view((*batch_shape, -1)).all(dim=-1, keepdim=True)
    return feasible_output


## Functions implementing CPU parallelization for the last layer bound computations.
def last_bounds_cpu_server(pid, bounds_net, servers_queue, instructions_queue, barrier):
    # Function implementing a CPU process computing last layer bounds (in parallel) until BaB termination is sent.
    try:
        while True:
            # Full synchronizatin after every batch.
            barrier.wait()

            comm = instructions_queue.get(True)  # blocking get on queue
            if len(comm) == 1:
                if comm[0] == "terminate":
                    break
                elif comm[0] == "idle":
                    continue
                else:
                    raise ChildProcessError(f"Message type not supported: {comm}")

            splitted_domain, splitted_lbs, splitted_ubs, slice_indices = comm
            splitted_lbs, splitted_ubs, dom_ub_point, dual_solutions = compute_last_bounds_sequentially(
                bounds_net, splitted_domain, splitted_lbs, splitted_ubs, slice_indices, share=True)

            # Send results to master
            servers_queue.put((splitted_lbs, splitted_ubs, dom_ub_point, dual_solutions, slice_indices))

    except Exception as e:
        # Exceptions from subprocesses are not caught otherwise.
        print(e)
        print(traceback.format_exc())


def spawn_cpu_servers(p, bounds_net):
    # Create child processes to parallelize the last layer bounds computations over cpu. Uses multiprocessing.
    servers_queue = mp.Queue()
    instruction_queue = mp.Queue()
    barrier = mp.Barrier(p)
    cpu_servers = mp.spawn(last_bounds_cpu_server,
                           args=(copy.deepcopy(bounds_net), servers_queue, instruction_queue, barrier), nprocs=(p-1),
                           join=False)
    return cpu_servers, servers_queue, instruction_queue, barrier


def join_children(gurobi_dict, timeout):
    cpu_servers = gurobi_dict["cpu_servers"]
    barrier = gurobi_dict["barrier"]
    instruction_queue = gurobi_dict["instruction_queue"]
    gurobi = gurobi_dict["gurobi"]
    p = gurobi_dict["p"]

    if gurobi and p > 1:
        # terminate the cpu servers and wait for it.
        barrier.wait()
        for _ in range(p):
            instruction_queue.put(("terminate",))
        cpu_servers.join(timeout=timeout)


def subproblems_to_cpu(splitted_domain, splitted_lbs, splitted_ubs, share=False, squeeze_interm=False):
    # Copy domain and bounds over to the cpu, sharing their memory in order for multiprocessing to access them directly.
    cpu_splitted_domain = splitted_domain.cpu()
    cpu_splitted_domain.share_memory_()
    cpu_splitted_lbs = []
    cpu_splitted_ubs = []
    for lbs, ubs in zip(splitted_lbs, splitted_ubs):
        cpu_lbs = lbs.cpu()
        cpu_ubs = ubs.cpu()
        if squeeze_interm:
            cpu_lbs = cpu_lbs.squeeze(0)
            cpu_ubs = cpu_ubs.squeeze(0)
        if share:
            cpu_lbs.share_memory_()
            cpu_ubs.share_memory_()
        cpu_splitted_lbs.append(cpu_lbs)
        cpu_splitted_ubs.append(cpu_ubs)
    return cpu_splitted_domain, cpu_splitted_lbs, cpu_splitted_ubs


def share_tensors(tensors):
    # Put a (list of) tensor(s) in shared memory. Copies to CPU in case it wasn't there.
    if isinstance(tensors, list):
        for i in range(len(tensors)):
            tensors[i] = tensors[i].cpu().share_memory_()
    else:
        tensors = tensors.cpu().share_memory_()
    return tensors


def add_terms(dest, orig, idx):
    # add information at the index idx from source to the destination 
    # dest is lists of tensor items and idx is an integer
    for item_idx in range(len(orig)):
       dest[item_idx].append(orig[item_idx][2*idx])
       dest[item_idx].append(orig[item_idx][2*idx+1])


def add_terms_lists(dest, orig, idx):
    # add information at the index idx from source to the destination 
    # dest is lists of lists and idx is an integer
    for item_idx in range(len(orig)):
        list_len = len(orig[item_idx])
        if len(dest[item_idx])==0:
            dest[item_idx] += [[] for _ in range(list_len)]
        for list_idx in range(list_len):
           dest[item_idx][list_idx].append(orig[item_idx][list_idx][2*idx])
           dest[item_idx][list_idx].append(orig[item_idx][list_idx][2*idx+1])

def modify_forms(items, lists):
    # modify the format of input items so they are consistent 
    # with following operations
    # we only do torch stack now, because stack is expensive according to Lewis.
    # we use lists instead until the final step.
    final_results = []
    for item in items:
        item = torch.stack(item)
        final_results.append(item)
    for ls in lists:
        ls_temp = []
        for sub_ls in ls:
            sub_ls = torch.stack(sub_ls)
            ls_temp.append(sub_ls)
        final_results.append(ls_temp)
    return final_results
