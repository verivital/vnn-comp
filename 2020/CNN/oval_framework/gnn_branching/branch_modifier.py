import torch
import glob
import os
import numpy as np
import shutil
import copy

'''
this script is used to modify branch files to add or delete certain features
to best accomodate training requirements

Important Features:
    1. add label --- for forming verfication layer
    2. add threshold criteria
'''



def branch_modify(path, y_label=False, relu_terms=False, dataset='cifar'):
    # load branches
    path = path+'*'
    all_branches = glob.glob(path)
    target_counter = 0
    total = len(all_branches)
    print(f'total number of branches: {len(all_branches)}')
    effective_points = []
    redundant_branch_counter=0
    #import pdb; pdb.set_trace()
    if y_label:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        if dataset == 'cifar':
            test = datasets.CIFAR10('./cifardata/', train=False, transform=transforms.ToTensor())
        elif dataset == 'mnist':
            test = datasets.MNIST("/home/jodie/PLNN/PLNN-verification-private/data", train=False, download=True, transform =transforms.ToTensor())
        
    for gt in all_branches:
        dc = torch.load(gt, map_location=torch.device('cpu'))

        if y_label:
            # adding label for Y category
            idx = int(gt.split('_')[-8])
            _,y = test[idx]
            try:
                y =y.item()
            except:
                pass
            prop = int(gt.split('_')[-6])
            assert prop != y
            dc['Y']['label'] = f'pred_{y}_against_{prop}'

        if relu_terms:
            decision = dc['Y']['decision']
            mask = dc['X']['mask']
            mask = [i.view(-1) for i in mask]
            dc['X']['mask'] = mask
            
            new_gl_lower = dc['relu']['true_lp_bounds'][decision[0]][decision[1]]
            orig_gl_lower = dc['X']['global_lower_bound'][0][0]
            improvement = min(new_gl_lower[0],0) + min(new_gl_lower[1],0) - 2*orig_gl_lower
            
            dc['relu']['rel_dis'] = {}
            eff_point_branch=0
            best_target_counter=0
            eff_target_counter = 0
            for layer in dc['relu']['true_lp_bounds'].keys():
                dc['relu']['rel_dis'][layer]={}
                for idx in dc['relu']['true_lp_bounds'][layer].keys():
                    lbs = dc['relu']['true_lp_bounds'][layer][idx]
                    #rdo_1 = (-lbs[0].item() + orig_gl_lower)/orig_gl_lower
                    #rdo_1 = round(rdo_1,4)
                    #rdo_2 = (-lbs[1].item() + orig_gl_lower)/orig_gl_lower
                    #rdo_2 = round(rdo_2,4)
                    rel_temp  = (min(lbs[0].item(),0) +min(lbs[1].item(),0) -2*orig_gl_lower)/improvement
                    if rel_temp> 0.90:
                        eff_target_counter += 1
                    if rel_temp==1.:
                        best_target_counter += 1
                    #current_branch[f'relu_{layer}_{idx}'] = [rdo_1, rdo_2,rdn_1, rdn_2]
                    if mask[layer][idx] == -1:
                        eff_point_branch += 1
                        dc['relu']['rel_dis'][layer][idx] = rel_temp
            print(f'current branch effective decision point number: {eff_point_branch}')
            print(f'current branch true decision point number: {eff_target_counter}')
            if best_target_counter == eff_point_branch:
                print('all points are good points ==> move the branch to a redundant folder')
                #import pdb; pdb.set_trace()
                shutil.move(gt, './cifar_kw_prox_m2_train_data/redundant/.')
                redundant_branch_counter += 1
                continue

            else:
                effective_points.append(eff_target_counter)
            #import pdb; pdb.set_trace()


            temp = [torch.zeros(len(i)) for i in mask]
            mask_temp = [torch.zeros(len(i)) for i in mask]
            for layer in dc['relu']['rel_dis'].keys():
                for idx in dc['relu']['rel_dis'][layer].keys():
                    temp[layer][idx]= dc['relu']['rel_dis'][layer][idx]
                    mask_temp[layer][idx] = 1
            decision = torch.abs(torch.cat([i for i in temp],0))
            mask_rel_1d = torch.cat([i for i in mask_temp], 0)
            dc['relu']['rel_mask_1d'] = mask_rel_1d
            dc['relu']['rel_decision'] = decision
            if max(decision) < 0.999 or max(decision)>1.001:
                print(gt)
                import pdb; pdb.set_trace()

        #import pdb; pdb.set_trace()
        torch.save(dc, gt)
        target_counter += 1
        print(f'progress: {target_counter}/{total}')

    #print(f'overall eff target ratio: {eff_target_counter/total}')
    print(f'total number of redundant branches: {redundant_branch_counter}')
    #import pdb; pdb.set_trace()
    return



################################### stats and check functions



def kw_score_check(path):
    from tools.bab_tools.model_utils import load_cifar_1to1_exp
    from plnn.proxlp_solver.solver import SaddleLP
    from plnn.branch_and_bound.branching_kw_score import choose_node_conv
    # load branches
    path = path+'*'
    all_branches = glob.glob(path)
    target_counter = 0
    total = len(all_branches)
    print(f'total number of branches: {len(all_branches)}')
    icp_score = 0
    random_order = list(range(3))
    sparsest_layer=0
    try:
        random_order.remove(sparsest_layer)
        random_order = [sparsest_layer] + random_order
    except:
        pass
    
    score = []
    kw_notin = 0
    for gt in all_branches:
        # load model
        imag_idx = int(gt.split('_')[-8])
        prop = int(gt.split('_')[-6])
        eps_temp= float(gt.split('_')[-4])
        x, verif_layers, test = load_cifar_1to1_exp('cifar_base_kw', imag_idx, prop)
        cuda_verif_layers = [copy.deepcopy(lay).cuda() for lay in verif_layers]
        intermediate_net = SaddleLP(cuda_verif_layers)
        domain = torch.stack([x.squeeze(0) - eps_temp, x.squeeze(0) + eps_temp], dim=-1)
        domain = domain.cuda()
        intermediate_net.define_linear_approximation(domain.unsqueeze(0))

        dc = torch.load(gt, map_location=torch.device('cpu'))
        mask = dc['X']['mask']
        mask = [i.to('cuda') for i in mask]
        orig_lbs = dc['X']['lower_bounds']
        orig_lbs = [i.to('cuda') for i in orig_lbs]
        orig_ubs = dc['X']['upper_bounds']
        orig_ubs = [i.to('cuda') for i in orig_ubs]
        branching_decision, icp_score, _ = choose_node_conv(orig_lbs, orig_ubs, mask, intermediate_net.weights,icp_score, random_order, sparsest_layer)
        try:
            score.append(dc['relu']['rel_dis'][branching_decision[0]][branching_decision[1]].item())
        except:
            kw_notin += 1
    
    import pdb;pdb.set_trace()


if __name__ == "__main__":
    path = './cifar_kw_prox_m2_train_data/first_attempt/trainfull_sub2/'
    #path = './cifar_kw_prox_m2_train_data/fake_attempt/val_fake_props/'
    branch_modify(path, y_label=True, relu_terms=True, dataset='cifar')
    #kw_score_check(path)
