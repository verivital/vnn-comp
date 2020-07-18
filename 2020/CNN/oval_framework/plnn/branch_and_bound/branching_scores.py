import torch
from torch import nn
from torch.nn import functional as F
from plnn.proxlp_solver.utils import LinearOp, ConvOp, BatchConvOp, BatchLinearOp
from plnn.modules import Flatten
from gnn_branching.graph_conv import GraphNet 
import time

class BranchingChoice:

    def __init__(self, init_mask, sparsest_layer, net_weights, branching_net_name, lr=1e-4, wd=1e-4, online=False):
        random_order = list(range(len(init_mask)))
        try:
            random_order.remove(sparsest_layer)
            random_order = [sparsest_layer] + random_order
        except:
            pass
        self.random_order = random_order
        self.icp_score_counter = 0
        self.icp_score_counter_seq = 0
        self.infeasible_count = 0
        self.decision_threshold=0.001
        self.sparsest_layer = sparsest_layer

        # net_weights
        self.net_weights = net_weights

        if branching_net_name is None:
            print('Using heuristic branching choices')
        else:
            print('Using gnn branching choices')
            model =  GraphNet(2, 64)

            model.load_state_dict(torch.load(branching_net_name))
            model.eval()
            self.model = model.cuda()
            if online:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay = wd)

            self.online = online
            trans_len = []
            temp = 0
            for i in init_mask:
                temp += len(i.view(-1))
                trans_len.append(temp)
            self.trans_len = torch.tensor(trans_len)
            self.layers = {}
            fixed_layers = [] 
            for idx, layer in enumerate(self.net_weights[:-1]):
                fixed_layers.append(layer)
                fixed_layers.append(nn.ReLU())
                if type(layer) in [BatchConvOp, ConvOp] and type(self.net_weights[idx+1]) is LinearOp:
                    fixed_layers.append(Flatten())

            self.layers['fixed_layers'] = fixed_layers
            ## LinearOp and BatchLinearOp use weights attribute for
            ## nn class weight attribute. The inconsistence will lead to
            ## a considerable amount of additin of condition clauses. To
            ## simplify the issue, we introduce a new weight attribute for
            ## the final property layer.
            self.net_weights[-1].weight = self.net_weights[-1].weights
            self.layers['prop_layers'] = [self.net_weights[-1]]


    def heuristic_branching_decision(self, lower_bounds_stacks, upper_bounds_stacks, orig_mask_stacks):
        '''
        choose the dimension to split on
        based on each node's contribution to the cost function
        in the KW formulation.
        net_weights are the weights as processed by plnn.proxlp_solver.solver.SaddleLP

        sparsest_layer: if all layers are dense, set it to -1
        decision_threshold: if the maximum score is below the threshold,
                            we consider it to be non-informative
        random_order: priority to each layer when making a random choice
                      with preferences. Increased preference for later elements in the list

        '''

        ## transform all terms into list of batches
        batch_size = len(lower_bounds_stacks)
        mask_temp = list(zip(*orig_mask_stacks))
        mask = [torch.stack(i,0) for i in mask_temp]
        mask = [(i == -1).float().view(batch_size, -1) for i in mask]
        lower_bounds_temp = list(zip(*lower_bounds_stacks))
        lower_bounds = [torch.stack(i, 0) for i in lower_bounds_temp]
        upper_bounds_temp = list(zip(*upper_bounds_stacks))
        upper_bounds = [torch.stack(i,0) for i in upper_bounds_temp]
        score = []
        intercept_tb = []

        ratio = torch.ones((batch_size,1,1), device=lower_bounds[0].device)
        # starting from 1, back-propogating: if the weight is negative
        # introduce bias; otherwise, intercept is 0
        # we are only interested in two terms for now: the slope x bias of the node
        # and bias x the amount of argumentation introduced by later layers.
        # From the last relu-containing layer to the first relu-containing layer

        for x_idx, layer in reversed(list(enumerate(self.net_weights))):
            if x_idx > 0:
                ratio = layer.backward(ratio)
                # assert ratio.shape[0] == 1

                # compute KW ratio
                ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[x_idx],
                                                           upper_bounds[x_idx])
                # Intercept
                intercept_temp = torch.clamp(ratio, max=0)
                intercept_candidate = intercept_temp * ratio_temp_1
                intercept_tb.insert(0, intercept_candidate.view(batch_size, -1) * mask[x_idx-1])

                # Bias
                b_temp = self.net_weights[x_idx-1].bias.detach()
                if type(self.net_weights[x_idx-1]) in [BatchConvOp, ConvOp]:
                    if type(self.net_weights[x_idx-1]) is BatchConvOp:
                        b_temp = self.net_weights[x_idx-1].unconditioned_bias.detach()
                    b_temp = b_temp.view(*((1,) * (ratio_temp_0.dim() - 3)), *b_temp.shape)
                ratio_1 = ratio * (ratio_temp_0 - 1)
                bias_candidate_1 = b_temp * ratio_1
                ratio = ratio * ratio_temp_0
                bias_candidate_2 = b_temp * ratio
                bias_candidate = torch.max(bias_candidate_1, bias_candidate_2)
                # test = (intercept_candidate!=0).float()
                # ???if the intercept_candiate at a node is 0, we should skip this node
                #    (intuitively no relaxation triangle is introduced at this node)
                #    score_candidate = test*bias_candidate + intercept_candidate
                score_candidate = bias_candidate + intercept_candidate
                score.insert(0, abs(score_candidate).view(batch_size, -1) * mask[x_idx-1])

        #import pdb; pdb.set_trace()
        # the following is not parallelised 
        decision_list = []
        for idx in range(batch_size):
            random_choice = self.random_order.copy()
            mask_item = [m[idx] for m in mask]
            score_item = [s[idx] for s in score]
            max_info = [torch.max(i, 0) for i in score_item]
            decision_layer = max_info.index(max(max_info))
            decision_index = max_info[decision_layer][1].item()
            if decision_layer != self.sparsest_layer and max_info[decision_layer][0].item() > self.decision_threshold:
                # temp = torch.zeros(score_item[decision_layer].size())
                # temp[decision_index]=1
                # decision_index = torch.nonzero(temp.reshape(mask[decision_layer].shape))[0].tolist()
                decision = [decision_layer, decision_index]

            else:
                intercept_tb_item = [i_tb[idx] for i_tb in intercept_tb]
                min_info = [[i, torch.min(intercept_tb_item[i], 0)] for i in range(len(intercept_tb_item)) if
                            torch.min(intercept_tb_item[i]) < -1e-4]
                # import pdb; pdb.set_trace()
                if len(min_info) != 0 and self.icp_score_counter < 2:
                    intercept_layer = min_info[-1][0]
                    intercept_index = min_info[-1][1][1].item()
                    self.icp_score_counter += 1
                    # inter_temp = torch.zeros(intercept_tb[intercept_layer].size())
                    # inter_temp[intercept_index]=1
                    # intercept_index = torch.nonzero(inter_temp.reshape(mask[intercept_layer].shape))[0].tolist()
                    decision = [intercept_layer, intercept_index]
                    if intercept_layer != 0:
                        self.icp_score_counter = 0
                    print('\tusing intercept score')
                else:
                    print('\t using a random choice')
                    undecided = True
                    while undecided:
                        preferred_layer = random_choice.pop(-1)
                        if len(mask_item[preferred_layer].nonzero()) != 0:
                            decision = [preferred_layer, mask_item[preferred_layer].nonzero()[0].item()]
                            undecided = False
                        else:
                            pass
                    self.icp_score_counter = 0
            decision_list.append(decision)
        return decision_list, score

    # TO BE REMOVED
    # Currently kept for debugging purpose
    def heuristic_sequential(self, lower_bounds, upper_bounds, orig_mask):
        '''
        choose the dimension to split on
        based on each node's contribution to the cost function
        in the KW formulation.
        net_weights are the weights as processed by plnn.proxlp_solver.solver.SaddleLP

        sparsest_layer: if all layers are dense, set it to -1
        decision_threshold: if the maximum score is below the threshold,
                            we consider it to be non-informative
        random_order: priority to each layer when making a random choice
                      with preferences. Increased preference for later elements in the list

        '''

        mask = [(i == -1).float().view(-1) for i in orig_mask]
        score = []
        intercept_tb = []
        random_choice = self.random_order.copy()

        ratio = torch.ones((1,1,1), device=lower_bounds[0].device)
        # starting from 1, back-propogating: if the weight is negative
        # introduce bias; otherwise, intercept is 0
        # we are only interested in two terms for now: the slope x bias of the node
        # and bias x the amount of argumentation introduced by later layers.
        # From the last relu-containing layer to the first relu-containing layer

        # Record score in a dic
        # new_score = {}
        # new_intercept = {}

        for x_idx, layer in reversed(list(enumerate(self.net_weights))):
            if x_idx > 0:
                ratio = layer.backward(ratio)
                # assert ratio.shape[0] == 1

                # compute KW ratio
                ratio_temp_0, ratio_temp_1 = compute_ratio(lower_bounds[x_idx],
                                                           upper_bounds[x_idx])
                # Intercept
                intercept_temp = torch.clamp(ratio, max=0)
                intercept_candidate = intercept_temp * ratio_temp_1
                intercept_tb.insert(0, intercept_candidate.view(-1) * mask[x_idx-1])

                # Bias
                b_temp = self.net_weights[x_idx-1].bias.detach()
                if type(self.net_weights[x_idx-1]) in [BatchConvOp, ConvOp]:
                    if type(self.net_weights[x_idx-1]) is BatchConvOp:
                        b_temp = self.net_weights[x_idx-1].unconditioned_bias.detach()
                    b_temp = b_temp.view(*((1,) * (ratio_temp_0.dim() - 3)), *b_temp.shape)
                ratio_1 = ratio * (ratio_temp_0 - 1)
                bias_candidate_1 = b_temp * ratio_1
                ratio = ratio * ratio_temp_0
                bias_candidate_2 = b_temp * ratio
                bias_candidate = torch.max(bias_candidate_1, bias_candidate_2)
                # test = (intercept_candidate!=0).float()
                # ???if the intercept_candiate at a node is 0, we should skip this node
                #    (intuitively no relaxation triangle is introduced at this node)
                #    score_candidate = test*bias_candidate + intercept_candidate
                score_candidate = bias_candidate + intercept_candidate
                score.insert(0, abs(score_candidate).view(-1) * mask[x_idx-1])

        max_info = [torch.max(i, 0) for i in score]
        decision_layer = max_info.index(max(max_info))
        decision_index = max_info[decision_layer][1].item()
        if decision_layer != self.sparsest_layer and max_info[decision_layer][0].item() > self.decision_threshold:
            # temp = torch.zeros(score[decision_layer].size())
            # temp[decision_index]=1
            # decision_index = torch.nonzero(temp.reshape(mask[decision_layer].shape))[0].tolist()
            decision = [decision_layer, decision_index]

        else:
            min_info = [[i, torch.min(intercept_tb[i], 0)] for i in range(len(intercept_tb)) if
                        torch.min(intercept_tb[i]) < -1e-4]
            # import pdb; pdb.set_trace()
            if len(min_info) != 0 and self.icp_score_counter_seq < 2:
                intercept_layer = min_info[-1][0]
                intercept_index = min_info[-1][1][1].item()
                self.icp_score_counter_seq += 1
                # inter_temp = torch.zeros(intercept_tb[intercept_layer].size())
                # inter_temp[intercept_index]=1
                # intercept_index = torch.nonzero(inter_temp.reshape(mask[intercept_layer].shape))[0].tolist()
                decision = [intercept_layer, intercept_index]
                if intercept_layer != 0:
                    self.icp_score_counter_seq = 0
                print('\tusing intercept score')
            else:
                print('\t using a random choice')
                undecided = True
                while undecided:
                    preferred_layer = random_choice.pop(-1)
                    if len(mask[preferred_layer].nonzero()) != 0:
                        decision = [preferred_layer, mask[preferred_layer].nonzero()[0].item()]
                        undecided = False
                    else:
                        pass
                self.icp_score_counter_seq = 0
        return decision, score


    def gnn_branching_decision(self, lower_bounds_stacks, upper_bounds_stacks, dual_vars_stacks, primal_input_stacks, primals_stacks, orig_mask_stacks):
        batch_size = len(lower_bounds_stacks)
        mask_temp = list(zip(*orig_mask_stacks))
        mask = [torch.stack(i,0) for i in mask_temp]
        mask = [(i == -1).float().view(batch_size, -1) for i in mask]
        mask_1d = torch.cat(mask, 1)
        lower_bounds_temp = list(zip(*lower_bounds_stacks))
        lower_bounds = [torch.cat(i, 0) for i in lower_bounds_temp]
        upper_bounds_temp = list(zip(*upper_bounds_stacks))
        upper_bounds = [torch.cat(i,0) for i in upper_bounds_temp]
        dual_temp = list(zip(*dual_vars_stacks))
        dual_vars = [torch.cat(i,0) for i in dual_temp]
        primal_inputs = torch.cat(primal_input_stacks, 0)
        primals = [torch.cat(primals_stacks,0)]
        self.layers['prop_layers'] = [self.layers['prop_layers'][0]]*batch_size

        start = time.time()

        scores = self.model(lower_bounds, upper_bounds, dual_vars, primals, primal_inputs, self.layers, mask_1d)
        end = time.time()
        print(f'graph requires: {end-start}')
            
        decision_list = []
        for s_idx in range(len(scores)):
            gnn_score, choice = torch.max(scores[s_idx],0)
            idx = (mask_1d[s_idx].nonzero()[choice]).item()
            dec_lay =  ((self.trans_len>idx).nonzero()[0]).item()
            if dec_lay == 0:
                dec_idx = idx
            else:
                dec_idx = (idx - self.trans_len[dec_lay-1]).item()
            decision_list += [[dec_lay, dec_idx]]

        if self.online:
            self.scores = scores
            self.mask_1d = mask_1d
            self.gnn_score = gnn_score

        return decision_list


    def online_learning(self, kw_decision, improvement):
        if kw_decision[0] == 0:
            partial_len = 0
        else:
            partial_len = self.trans_len[kw_decision[0]-1]
        print('updating the trained model')
        partial_len = partial_len + kw_decision[1]
        kw_index = len(self.mask_1d[0][:partial_len].nonzero())
        kw_score = self.scores[0][kw_index]
        #assert kw_score == self.new_score[kw_decision[0]][kw_decision[1]]
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.gnn_score - kw_score + improvement
        loss.backward()
        self.optimizer.step()
        self.model.eval()

    def del_score(self):
        del self.scores
        del self.gnn_score
        del self.mask_1d
        #del self.new_score



def compute_ratio(lower_bound, upper_bound):
    lower_temp = lower_bound - F.relu(lower_bound)
    upper_temp = F.relu(upper_bound)
    slope_ratio = upper_temp / (upper_temp - lower_temp)
    intercept = -1 * lower_temp * slope_ratio

    return slope_ratio, intercept
