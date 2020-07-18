import torch
from tools.bab_tools.model_utils import cifar_model_m2, add_single_prop


def load_cifar_1to1_layers_dc(model):
    '''
    return a dictionary of fixed network layers and 
    final layers incorporating different properties
    '''
    if model=='cifar_base_kw':
        model_name = './models/cifar_base_kw.pth'
        model = cifar_model_m2()
        model.load_state_dict(torch.load(model_name, map_location = "cpu")['state_dict'][0])
    else:
        raise NotImplementedError

    layers_dc = {}
    
    layers = list(model.children())
    layers_dc['fixed_layers'] = layers[:-1]
    for y_pred in range(10):
        for test in range(10):
            if y_pred == test:
                continue
            else:
                added_prop_layers = add_single_prop(layers, y_pred, test)
                layers_dc[f'pred_{y_pred}_against_{test}'] = added_prop_layers[-1]
    return layers_dc



def process_branches(branches, linearize=False):
    
    # collecting all necessory information
    # each sublist contains lower_bounds_all, upper_bounds_all, layers, decision
    graph_info_all = []
    for branch in branches:
        info_mat = {}
        dc = torch.load(branch, map_location=torch.device('cpu'))

        rel_dec = dc['relu']['rel_decision']
        rel_mask = dc['relu']['rel_mask_1d']
        rel_dec_selec = rel_dec[rel_mask.nonzero()]
        if min(rel_dec_selec) > 0.95:
            continue

        info_mat['rel_decision'] = dc['relu']['rel_decision']

        if linearize:
            raise NotImplementedError
        else:
            info_mat['lower_bounds'] = [i.squeeze(0).data for i in dc['X']['lower_bounds']]
            info_mat['upper_bounds'] =[i.squeeze(0).data for i in dc['X']['upper_bounds']]


        #info_mat.append([x.data.view(-1) for x in dc['X']['upper_bounds']])
        # get layers
        info_mat['label'] = dc['Y']['label']

        
        #mask = [(i==-1).float() for i in mask]
        #mask_1d = th.cat([i for i in mask],0)
        #info_mat.append(mask_1d)

        info_mat['rel_mask_1d'] = dc['relu']['rel_mask_1d']
        
        #primal_dual info
        dual_vars_list = [i.squeeze(0).view(2,-1).T for i in dc['gb']['dual_vars']]
        info_mat['dual_vars'] = dual_vars_list
        info_mat['primal_input'] = dc['gb']['primal_input']
        #info_mat['primals'] = [torch.tensor(i) for i in dc['gb']['primals']]
        info_mat['primals'] = [dc['X']['global_lower_bound'][0]]

        graph_info_all.append(info_mat)

    return graph_info_all


def hinge_rank_loss(output, target, thres_max):
    # since we need to expand the size of output and target to its square
    #to compute the rank loss, we limit the max batch size to be 50
    '''
    TODO: CHECK THE CODE. INITIALLY, THERE IS A PDB BREAKPOINT
    '''

    max_batch = 50
    if len(target)%max_batch == 0:
        increment = 0
    else:
        increment = 1
    batch_number = len(target)//max_batch + increment
    loss_sum = torch.zeros(1).cuda()
    total_number = len(target)
    for idx in range(batch_number):
        output_batch = output[idx*max_batch:(idx+1)*max_batch] 
        target_batch = target[idx*max_batch:(idx+1)*max_batch] 
        loss_batch = torch.clamp(1.0 - output_batch.unsqueeze(2).expand(output_batch.size(0), output_batch.size(1), output_batch.size(1)) + output_batch.unsqueeze(1), min=0)
        target_batch =  target_batch.unsqueeze(2).expand(target_batch.size(0), target_batch.size(1), target_batch.size(1)) - target_batch.unsqueeze(1)
        target_batch = ((target_batch>0).float() *(target_batch<thres_max).float()).cuda()
        loss_batch = loss_batch * target_batch
        target_sum = torch.sum(target_batch, (1,2))
        invalid_cases = (target_sum==0).float()
        total_number = total_number - int(torch.sum(invalid_cases).item())
        #import pdb; pdb.set_trace()
        temp_sum = torch.sum(loss_batch, (1,2))/(target_sum + invalid_cases.cuda())
        loss_sum = loss_sum + torch.sum(temp_sum)
    if total_number!=0:
        loss = loss_sum/total_number
    else:
        loss = loss_sum
    return loss
    #output = output.view(-1)
    #P = (target==1).nonzero().flatten()
    #N = (target!=1).nonzero().flatten()
    #score_n = output[N].repeat(len(P))
    #score_p = (output[P].unsqueeze(-1)).repeat(1,len(N)).view(-1)
    #loss = torch.clamp(1 + score_n -score_p, min=0).sum()

    #return loss/(P.numel()*N.numel())






def hinge_rank_loss_weighted(output, target, rel_dis):
    output = output.view(-1)
    P = (target==1).nonzero().flatten()
    N = (target!=1).nonzero().flatten()
    score_n = output[N].repeat(len(P))
    weight = (1-rel_dis[N]).repeat(len(P))
    score_p = (output[P].unsqueeze(-1)).repeat(1,len(N)).view(-1)
    loss = torch.clamp(1 + score_n -score_p, min=0)
    loss = (weight*loss).sum()

    return loss/(P.numel()*N.numel())
    


def add_weights_to_loss(target, loss, N=None):
    weight = target.clone()
    w1 = (1.*len(weight))/len(weight.nonzero().view(-1))
    w2 = (1.*len(weight))/(len(weight)-len(weight.nonzero().view(-1)))
    weight = weight*(w1-w2) + w2
    loss = loss * weight
    if N is None:
        return torch.mean(loss)
    else:
        return loss/N


    #sigm = nn.Sigmoid()
    #criterion = nn.BCELoss(reduction = 'none')
    #criterion = nn.MSELoss('reduction = none')
    #criterion = nn.HingeEmbeddingLoss(margin=0.1, reduction='none')

def BCElogitloss(output, target, weight = False):
    if weight is False:
        loss_fn = torch.nn.BCEWithLogitsLoss()
        return loss_fn(output, target.unsqueeze(-1))
    else:
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction = 'none')
        loss = loss_fn(output, target.unsqueeze(-1))
        weight = target.clone()
        w1 = (1.*len(weight))/len(weight.nonzero().view(-1))
        w2 = (1.*len(weight))/(len(weight)-len(weight.nonzero().view(-1)))
        weight = weight*(w1-w2) + w2
        loss = loss * weight.unsqueeze(-1)
        return torch.mean(loss)

