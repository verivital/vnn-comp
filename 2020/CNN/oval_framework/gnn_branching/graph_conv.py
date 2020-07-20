#! /usr/bin/env python
import pickle
import os
import glob
import shutil
import argparse
import torch as th
import numpy as np
from torch import nn
from torch.nn import functional as F
from plnn.modules import Flatten
from plnn.proxlp_solver.utils import LinearOp, ConvOp, BatchConvOp, BatchLinearOp
import random

'''
Training test file for the deep model transferrability studies
1. added norm weight in the backward pass
2. has been tested on wide and deep networks --- satisfactory performances

Currently treated as the correct model 
'''

class EmbedLayerUpdate(nn.Module):
    '''
    this class updates embeded layer one time
    '''
    def __init__(self,p, T):
        super(EmbedLayerUpdate, self).__init__()
        '''
        p: embedding size
        T: number of passes
        '''
        self.p = p
        self.T = T

        #inputs
        self.inp_f = nn.Linear(3,p)
        self.inp_f_1 = nn.Linear(p,p)

        self.inp_b = nn.Linear(2,p)
        self.inp_b_1 = nn.Linear(p,p)
        self.inp_b2 = nn.Linear(2*p,p)
        self.inp_b2_2 = nn.Linear(p,p)
        #self.inp_b = nn.Linear(p, p)

        # for bound features of the node
        self.fc1 = nn.Linear(6, p)      
        self.fc1_1 = nn.Linear(p, p)      
        #self.fc1_2 = nn.Linear(p, p)      
        #self.fc2 = nn.Linear(3*p, p)
        #self.fc2_1 = nn.Linear(p, p)

        # for sum of neighbour embeddings vectors
        self.fc3 = nn.Linear(2*p, p)     
        self.fc3_2 = nn.Linear(p, p)     
        # for weights of edges connecting to the node. 
        # edge updates
        self.fc4 = nn.Linear(2*p,p)
        self.fc4_2 = nn.Linear(p,p)

        # outputs
        self.out1 = nn.Linear(4, p)
        self.out2 = nn.Linear(2*p, p)
        self.out3 = nn.Linear(p, p)
        
        #backwards
        self.bc1 = nn.Linear(6, p)
        self.bc1_1 = nn.Linear(p, p)
        self.bc1_2 = nn.Linear(p, p)
        self.bc2 = nn.Linear(3*p, p)
        self.bc2_1 = nn.Linear(p, p)
        self.bc3 = nn.Linear(2*p, p)
        self.bc3_1 = nn.Linear(p, p)
        self.bc4 = nn.Linear(2*p, p)
        self.bc4_1 = nn.Linear(p, p)


    def forward(self, lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_inputs, layers, mu):

        # NOTE: All bounds should be at the same size as the layer outputs
        #       We have assumed that the last property layer is linear       

        batch_size =len(lower_bounds_all[0])
        p = self.p
        for i in range(self.T):
            ## FORWARD PASS
            # for the forward pass the first layer is not updated
            #print(th.cat([lower_bounds_all[0].unsqueeze(-1),upper_bounds_all[0].unsqueeze(-1)],1))

            # first, deal with the input layer
            if i==0:
                inp = th.cat([lower_bounds_all[0].view(-1).unsqueeze(-1),
                              primal_inputs.view(-1).unsqueeze(-1),
                              upper_bounds_all[0].view(-1).unsqueeze(-1)],1)
                temp = self.inp_f_1(F.relu(self.inp_f(inp)))
                mu[0] = temp.reshape(mu[0].size())

            relu_count_idx = 0
            
            #if type(layers[0]) is nn.Linear:
            #    out_features = [mu[0].size()[1], mu[0].size()[0]]
            #elif type(layers[0]) nn.Conv2d:
            #    pic_len = int((mu[0].size()[0])**0.5)
            #    out_features = [mu[0].size()[1], 1, pic_len, pic_len]
            out_features = [-1]+ th.tensor(lower_bounds_all[0][0].size()).tolist()

            idx = 0
            for layer_idx, layer in enumerate(layers['fixed_layers']):
                #print('count_idx: ', count_idx)
                #print('layer_idx: ', layer_idx)
                if type(layer) in [BatchConvOp, ConvOp, nn.Conv2d]:
                    if type(layer) in [BatchConvOp, ConvOp]:
                        layer_weight = layer.weights
                    else:
                        layer_weight = layer.weight

                    if type(layer) is BatchConvOp:
                        layer_bias = layer.unconditioned_bias.detach().view(-1)
                    elif type(layer) is ConvOp:
                        layer_bias = layer.bias.view(-1)
                    else:
                        layer_bias = layer.bias
                    #reshape 
                    mu_inp = th.cat([i for i in mu[relu_count_idx]], 1)
                    mu_inp = th.t(mu_inp).reshape(out_features)
                    nb_embeddings_pre = F.conv2d(mu_inp, layer_weight , bias=None,
                                            stride=layer.stride, padding=layer.padding, dilation=layer.dilation, groups=layer.groups)
                    # record and transfer back
                    out_features = th.tensor(nb_embeddings_pre.size()).tolist()
                    nb_embeddings_temp = nb_embeddings_pre.reshape(out_features[0],-1)
                    #import pdb; pdb.set_trace()
                    nb_embeddings_temp = th.cat([nb_embeddings_temp[i*p:(1+i)*p] for i in range(batch_size)], 1)
                    nb_embeddings_temp = th.t(nb_embeddings_temp)
                    pre_layer_bias = layer_bias.unsqueeze(1).expand(out_features[1],out_features[2]*out_features[3])
                    pre_layer_bias = pre_layer_bias.reshape(-1)
                    pre_layer_bias = pre_layer_bias.repeat(batch_size)
                    layer_lower_pre = lower_bounds_all[idx+1].view(-1)
                    layer_upper_pre = upper_bounds_all[idx+1].view(-1)
                    idx += 1
                    #import pdb; pdb.set_trace()

                elif type(layer) in [nn.Linear, LinearOp, BatchLinearOp]: 
                    if type(layer) in [LinearOp, BatchConvOp]:
                        layer_weight = layer.weights
                    else:
                        layer_weight = layer.weight
                    nb_embeddings_temp = layer_weight @ mu[relu_count_idx]
                    nb_embeddings_temp = th.cat([i for i in nb_embeddings_temp], 0)
                    pre_layer_bias = layer.bias.repeat(batch_size)
                    out_features = [-1, nb_embeddings_temp.size()[0]]
                    layer_lower_pre = lower_bounds_all[idx+1].view(-1)
                    layer_upper_pre = upper_bounds_all[idx+1].view(-1)
                    idx += 1

                elif type(layer) is nn.ReLU:
                # node features
                    #layer_lower = lower_bounds_all[idx+1].view(-1)
                    #layer_upper = upper_bounds_all[idx+1].view(-1)
                    #if (idx == len(lower_bounds_all)-1):
                    #    layer_lower_pre = layer_lower
                    #    layer_upper_pre = layer_upper
                    #else:
                    #layer_lower_pre = lower_bounds_all[idx].view(-1)
                    #layer_upper_pre = upper_bounds_all[idx].view(-1)
                    ratio_0, ratio_1, beta, ambi_mask = compute_ratio(layer_lower_pre, layer_upper_pre)
                    #import pdb; pdb.set_trace()

                    # measure relaxation
                    layer_n_bounds = th.cat([beta.unsqueeze(-1), 
                                             layer_lower_pre.unsqueeze(-1),
                                             layer_upper_pre.unsqueeze(-1), 
                                             dual_vars[relu_count_idx][:,0].unsqueeze(-1),
                                             dual_vars[relu_count_idx][:,1].unsqueeze(-1),
                                             pre_layer_bias.unsqueeze(-1)],1)
                    layer_relax_s1 = self.fc1_1(F.relu(self.fc1(layer_n_bounds)))
                    layer_relax = layer_relax_s1 * ambi_mask.unsqueeze(-1)

                    #import pdb; pdb.set_trace()
                    #print('layer relax forward: ', th.max(abs(layer_relax)))


                    # embedding vector updates

                    nb_embeddings_input = th.cat([nb_embeddings_temp*ratio_0.unsqueeze(-1), nb_embeddings_temp*ratio_1.unsqueeze(-1)],1)
                    layer_nb_embeddings = self.fc3_2(F.relu(self.fc3(nb_embeddings_input)))
                    #import pdb; pdb.set_trace()

                    #print('layer_nb_embeddings: ', th.max(abs(layer_nb_embeddings)))

                    # update all nodes in a layer
                    layer_input = th.cat([layer_relax, layer_nb_embeddings],dim=1)
                    layer_mu_new  = self.fc4_2(F.relu(self.fc4(layer_input)))
                    layer_mu_new = layer_mu_new*(ratio_0!=0).float().unsqueeze(-1)
                    relu_count_idx += 1
                    #import pdb; pdb.set_trace()
                    mu[relu_count_idx] = layer_mu_new.reshape(mu[relu_count_idx].size())


                    if (th.sum(th.isnan(layer_mu_new))!=0):
                        print('mu contains nan')
                        import pdb;pdb.set_trace()

                elif type(layer) is Flatten:
                    out_features = [-1] + th.tensor(lower_bounds_all[idx+1].size()).tolist() 
                    pass
                else:
                    raise NotImplementedError

            # property layer 
            # forward pass
            nb_embeddings_temp = [layers['prop_layers'][i].weight @ mu[relu_count_idx][i] for i in range(batch_size)]

            pre_layer_bias = [layers['prop_layers'][i].bias for i in range(batch_size)]
            nb_embeddings_temp = th.cat(nb_embeddings_temp, 0)
            pre_layer_bias = th.cat(pre_layer_bias,0)
            layer_lower_pre = lower_bounds_all[idx+1].view(-1)
            layer_upper_pre = upper_bounds_all[idx+1].view(-1)
            layer_n_bounds = th.cat([ layer_lower_pre.unsqueeze(-1), 
                                      layer_upper_pre.unsqueeze(-1), 
                                      primals[0].unsqueeze(-1),
                                      pre_layer_bias.unsqueeze(-1)],1)
            layer_relax_output = F.relu(self.out1(layer_n_bounds))
            layer_input = th.cat([layer_relax_output, nb_embeddings_temp],dim=1)
            layer_mu_new  = self.out3(F.relu(self.out2(layer_input)))
            relu_count_idx += 1
            mu[relu_count_idx] = layer_mu_new.reshape(mu[relu_count_idx].size())
            
            #backward pass

            ratio = lower_bounds_all[0].new_full(lower_bounds_all[-1].size(), fill_value=1.)
            ratio = [th.t(layers['prop_layers'][i].weight)@ratio[i] for i in range(batch_size)]
            ratio = th.stack(ratio,0)
            nor_weight = [layers['prop_layers'][i].weight for i in range(batch_size)]
            next_layer = 'prop'
            #import pdb; pdb.set_trace()
            

            for layer_idx, layer in reversed(list(enumerate(layers['fixed_layers']))):
                # for node features
                #print('count_idx: ', count_idx)
                #print('layer_idx: ', layer_idx)
                
                if type(layer) in [BatchConvOp, ConvOp, nn.Conv2d]:
                    if type(layer) in [BatchConvOp,ConvOp]:
                        layer_weight = layer.weights
                    else:
                        layer_weight = layer.weight

                    if type(layer) is BatchConvOp:
                        layer_bias = layer.unconditioned_bias.detach().view(-1)
                    elif type(layer) is ConvOp:
                        layer_bias = layer.bias.view(-1)
                    else:
                        layer_bias = layer.bias

                    ratio = F.conv_transpose2d(ratio, layer_weight, stride = layer.stride, padding = layer.padding)

                    #Embedding vectors
                    weight_size = layer_weight.size()
                    #norm = th.norm(layer.weight.reshape(weight_size[0], weight_size[1], -1), dim=2).unsqueeze(-1).unsqueeze(-1)

                    nor_weight = layer_weight
                    #nor_weight = layer.weight/norm
                    #temp_weight = layer.weight

                    next_layer = layer
                    idx -= 1

                elif type(layer) in [nn.Linear, LinearOp, BatchLinearOp]: 
                    if type(layer) in [LinearOp, BatchConvOp]:
                        layer_weight = layer.weights
                    else:
                        layer_weight = layer.weight
                    w_temp = layer_weight
                    ratio = F.linear(ratio, th.t(w_temp))
                    
                    nor_weight = layer_weight
                    #nor_weight = layer.weight/th.norm(layer.weight, dim=1).unsqueeze(-1)
                    #temp_weight = layer.weight
                    next_layer = 'Linear'
                    idx -= 1
                    


                elif type(layer) is nn.ReLU:

                    layer_lower_pre = lower_bounds_all[idx]
                    layer_upper_pre = upper_bounds_all[idx]
                    #layer_lower = lower_bounds_all[layer_idx+1].view(-1)
                    #layer_upper = upper_bounds_all[layer_idx+1].view(-1)
                    layer_lower_pre = layer_lower_pre.view(-1)
                    layer_upper_pre = layer_upper_pre.view(-1)
                    ratio_0, ratio_1, beta, ambi_mask = compute_ratio(layer_lower_pre, layer_upper_pre)

                    #Bias
                    #pre_layer_bias = layers['fixed_layers'][layer_idx-1].bias
                    pre_layer_temp = layers['fixed_layers'][layer_idx-1]
                    if type(pre_layer_temp) is BatchConvOp:
                        pre_layer_bias = pre_layer_temp.unconditioned_bias.detach().view(-1)
                    elif type(pre_layer_temp) is ConvOp:
                        pre_layer_bias = pre_layer_temp.bias.view(-1)
                    else:
                        pre_layer_bias = pre_layer_temp.bias

                    if type(layers['fixed_layers'][layer_idx-1]) in [nn.Conv2d, BatchConvOp, ConvOp]:
                        out_features = lower_bounds_all[idx].size()
                        pre_layer_bias = pre_layer_bias.unsqueeze(1).expand(out_features[1],out_features[2]*out_features[3])
                        pre_layer_bias = pre_layer_bias.reshape(-1)

                    pre_layer_bias = pre_layer_bias.repeat(batch_size)

                    # measure relaxation
                    layer_n_bounds = th.cat([layer_lower_pre.unsqueeze(-1), 
                                             layer_upper_pre.unsqueeze(-1),
                                             beta.view(-1).unsqueeze(-1),
                                             dual_vars[relu_count_idx-2][:,1].unsqueeze(-1),
                                             dual_vars[relu_count_idx-2][:,0].unsqueeze(-1),
                                             pre_layer_bias.unsqueeze(-1)],dim=1)

                                             #intercept_candidate.unsqueeze(-1),
                                             #bias_candidate_1,
                                             #bias_candidate_2,

                    layer_relax_s1 = self.bc1_2(F.relu(self.bc1_1(F.relu(self.bc1(layer_n_bounds)))))
                    #layer_relax_s1 = self.bc1_1(F.relu(self.bc1(layer_n_bounds)))
                    layer_relax_s2_input = th.cat([layer_relax_s1, 
                                                    layer_relax_s1*(-dual_vars[relu_count_idx-2][:,1].unsqueeze(-1)),
                                                    layer_relax_s1*(dual_vars[relu_count_idx-2][:,0].unsqueeze(-1))
                                                    ], dim=1)
                    layer_relax = self.bc2_1(F.relu(self.bc2(layer_relax_s2_input)))

                    layer_relax = layer_relax * (ambi_mask).unsqueeze(-1)

                    #print('layer relax backward: ', th.max(abs(layer_relax)))


                    # embedding vector updates
                    if type(next_layer) in [nn.Conv2d, ConvOp, BatchConvOp]:
                        ## reshape embedding vectors
                        out_features = [-1]+ th.tensor(lower_bounds_all[idx+1].size()).tolist()[1:]
                        mu_inp = th.cat([i for i in mu[relu_count_idx]], 1)
                        mu_inp = th.t(mu_inp).reshape(out_features)
                        nb_embeddings_pre = F.conv_transpose2d(mu_inp, nor_weight , bias=None,
                                            stride=next_layer.stride, padding=next_layer.padding, dilation=next_layer.dilation, groups=next_layer.groups)
                        _, _, w_inp, h_inp = mu_inp.size()
                        _, _, w_wgt, h_wgt = nor_weight.size()
                        freq_inp = th.ones([1,1,w_inp, h_inp]).cuda()
                        freq_weight = th.ones([1,1, w_wgt, h_wgt]).cuda()
                        freq = F.conv_transpose2d(freq_inp, freq_weight , bias=None,
                                            stride=next_layer.stride, padding=next_layer.padding, dilation=next_layer.dilation, groups=next_layer.groups)
                        nb_embeddings_pre = nb_embeddings_pre/freq

                        # record and transfer back
                        out_features = nb_embeddings_pre.size()
                        nb_embeddings = nb_embeddings_pre.reshape(out_features[0],-1)
                        nb_embeddings = th.cat([nb_embeddings[i*p:(1+i)*p] for i in range(batch_size)], 1)
                        nb_embeddings = th.t(nb_embeddings)

                    elif next_layer is 'Linear':
                        nb_embeddings = th.t(nor_weight).matmul(mu[relu_count_idx])
                        nb_embeddings = th.cat([i for i in nb_embeddings], 0)
                    
                    elif next_layer is 'prop':
                        nb_embeddings = [th.t(nor_weight[i])@ mu[relu_count_idx][i] for i in range(batch_size)]
                        nb_embeddings = th.cat(nb_embeddings, 0)
                    else:
                        raise NotImplementedError


                    nb_embeddings_0 = nb_embeddings*ratio_0.unsqueeze(-1) 
                    nb_embeddings_1 = nb_embeddings*ratio_1.unsqueeze(-1)


                    layer_embeddings_input = th.cat([nb_embeddings_0, nb_embeddings_1],dim=1)
                    layer_nb_embeddings = self.bc3_1(F.relu(self.bc3(layer_embeddings_input)))


                    if (th.sum(th.isnan(layer_nb_embeddings))!=0):
                        print('layer_nb_embedding contains nan')
                        pdb.set_trace()

                    # update all nodes in a layer
                    layer_input = th.cat([layer_relax, layer_nb_embeddings],dim=1)
                    layer_mu_new  = self.bc4_1(F.relu(self.bc4(layer_input)))

                    layer_mu_new = layer_mu_new*((ratio_0!=0).view(-1)).float().unsqueeze(-1)

                    mu[relu_count_idx-1] = layer_mu_new.reshape(mu[relu_count_idx-1].size())
                    relu_count_idx -= 1
                    #print('relu_count_idx: ', relu_count_idx)
                    #mu = layer_mu_new
                    #print('layer_mu_new: ', th.max(abs(layer_mu_new)))    
                    #pdb.set_trace()
                elif type(layer) is Flatten:
                    ratio = ratio.reshape(lower_bounds_all[idx].size())
                else:
                    raise NotImplementedError

                if layer_idx == 0:
                    if type(next_layer) in [nn.Conv2d, ConvOp, BatchConvOp]:
                        ## reshape embedding vectors
                        out_features = [-1]+ th.tensor(lower_bounds_all[idx+1].size()).tolist()[1:]
                        mu_inp = th.cat([i for i in mu[relu_count_idx]], 1)
                        mu_inp = th.t(mu_inp).reshape(out_features)
                        nb_embeddings_pre = F.conv_transpose2d(mu_inp, nor_weight , bias=None,
                                            stride=next_layer.stride, padding=next_layer.padding, dilation=next_layer.dilation, groups=next_layer.groups)
                        # record and transfer back
                        out_features = nb_embeddings_pre.size()
                        nb_embeddings = nb_embeddings_pre.reshape(out_features[0],-1)
                        nb_embeddings = th.cat([nb_embeddings[i*p:(1+i)*p] for i in range(batch_size)], 1)
                        nb_embeddings = th.t(nb_embeddings)

                    elif next_layer is 'Linear':
                        nb_embeddings = th.t(nor_weight).matmul(mu[relu_count_idx])
                        nb_embeddings = th.cat([i for i in nb_embeddings], 0)
                    else:
                        raise NotImplementedError
                    
                    inp = th.cat([lower_bounds_all[0].view(-1).unsqueeze(-1),
                                  upper_bounds_all[0].view(-1).unsqueeze(-1)],1)
                    inp_relax = self.inp_b_1(F.relu(self.inp_b(inp)))
                    domain_input = th.cat([inp_relax, nb_embeddings],dim=1)
                    domain_mu_new  = self.inp_b2_2(F.relu(self.inp_b2(domain_input)))
                    mu[relu_count_idx-1] = domain_mu_new.reshape(mu[relu_count_idx-1].size())
                    #print('last layer', relu_count_idx-1)
            
        return mu





class EmbedUpdates(nn.Module):
    '''
    this class updates embeding vectors from t=1 and t=T
    '''

    def __init__(self, T, p):
        '''
        p_list contains the input and output dimensions of embedding vectors for all layers
        len(p_list) = T+1
        '''
        super(EmbedUpdates, self).__init__()
        self.T = T
        self.p = p
        #self.p_list = p_list
        #assert len(p_list)==T+1
        #self.updates = [EmbedLayerUpdate(p_list[i], p_list[i+1]) for i in range(0,T)]
        self.update = EmbedLayerUpdate(p,T)


    def forward(self, lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_inputs, layers):
        #mu = lower_bounds_all[0].new_full((len(lower_bounds_all[0]), self.p), fill_value=0.0) 
        mu = init_mu(lower_bounds_all, self.p)
        mu = self.update( lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_inputs, layers, mu)
        return mu



class ComputeFinalScore(nn.Module):
    '''
    this class computes a final score for each node

    p: the dimension of embedding vectors at the final stage
    '''
    def __init__(self, p):
        super(ComputeFinalScore,self).__init__()
        self.p = p
        #self.fnode = nn.Linear(p,1)
        self.fnode = nn.Linear(p,p)
        self.fscore = nn.Linear(p, 1)
    
    def utils(self, mu):
        scores = []
        for layer in mu[1:-1]:
            scores_current_layer = mu[0].new_full((layer.size()[0],), fill_value=0.0) 
            scores.append(scores_current_layer)
        return scores


    def forward(self, mu, masks):
        #scores = self.utils(mu)
        scores = []
        for batch_idx in range(len(mu[0])):
            mu_temp = th.cat([i[batch_idx] for i in mu[1:-1]],dim=0)
            mu_temp = mu_temp[masks[batch_idx].nonzero().view(-1)]
            temp = self.fnode(mu_temp)
            score = self.fscore(F.relu(temp))
            scores.append(score.view(-1))
        #scores = nn.utils.rnn.pad_sequence(scores, batch_first=True)

        #for layer_idx in range(1, len(mu)-1):
        #    temp = self.fnode(mu[layer_idx])
        #    layer_score = self.fscore(F.relu(temp))
        #    scores[layer_idx-1] = layer_score
        
        #import db; pdb.set_trace()
        # scores = [i.squeeze(1) for i in scores]
        # reshape into 1 dimension
        # scores = th.cat([i for i in scores],0)
        # remove all invalid options
        # scores = scores[mask.nonzero()].view(-1)

        #scores = th.cat([i for i in scores],0)
        #scores = scores[mask.nonzero().view(-1)]
        #scores = scores/max(scores)
        #scores = self.fscore_3(F.relu(self.fscore_2(scores)))
        #transform into relative number
        return scores


class GraphNet(nn.Module):
    def __init__(self, T, p):
        super(GraphNet,self).__init__()
        self.EmbedUpdates = EmbedUpdates(T, p)
        self.ComputeFinalScore = ComputeFinalScore(p)

    def forward(self, lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_inputs, layers, masks):
        mu = self.EmbedUpdates(lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_inputs, layers)
        scores = self.ComputeFinalScore(mu, masks)

        return scores



def init_mu(lower_bounds_all, p):
    
    mu = []
    batch_size = len(lower_bounds_all[0])
    for i in lower_bounds_all:
        required_size = i[0].view(-1).size()
        mus_current_layer = lower_bounds_all[0].new_full((batch_size,required_size[0],p), fill_value=0.0) 
        mu.append(mus_current_layer)
    
    return mu


def compute_ratio(lower_bound, upper_bound):
    lower_temp = lower_bound - F.relu(lower_bound)
    upper_temp = F.relu(upper_bound)
    diff = upper_temp-lower_temp
    zero_ids = (diff==0).nonzero()
    if len(zero_ids)>0:
        if th.sum(upper_temp[zero_ids])==0:
            diff[zero_ids] = 1e-5
    slope_ratio0 = upper_temp/diff

    intercept = -1*lower_temp*slope_ratio0
    ambi_mask = (intercept>0).float()
    slope_ratio1 = (1 - 2*(slope_ratio0*ambi_mask))*ambi_mask + slope_ratio0
    #if (lower >= 0):
    #    decision = [1,0]
    #elif (upper <=0):
    #    decision = [0, 0]
    #else: 
    #    temp = upper/(upper - lower)
    #    decision = [temp, -temp*lower]
    #return decision
    return slope_ratio0, slope_ratio1, intercept, ambi_mask


