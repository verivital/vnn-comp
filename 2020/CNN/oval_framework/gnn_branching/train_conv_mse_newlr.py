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
import random
import logger
from gnn_branching.graph_conv import GraphNet
from training_utils import *
import time


'''
New learning rate scheduling: decrease the learning rate by 5 times, when the accuracy
plateaus for 10 epochs

added batch support 

added various loss support

'''


############### Train

def train(files_train, files_val, args, sub_exp):
    model = GraphNet(args.T, args.p)


#    if th.cuda.device_count() > 1
#        print('Use ', th.cuda.device_count(), 'GPUs!')
#        model = nn.DataParallel(model)
    model.cuda()
    
    sigm = nn.Sigmoid()
    #criterion = nn.BCELoss(reduction = 'none')
    criterion = nn.MSELoss(reduction = 'sum')
    #criterion = nn.HingeEmbeddingLoss(margin=0.1, reduction='none')
    optimizer = th.optim.Adam(model.parameters(), lr = args.start_lr, weight_decay = args.wd)
    #optimizer = th.optim.Adam(model.parameters(), lr = args.lr, amsgrad = True)
    #optimizer = th.optim.Adagrad(model.parameters(), lr = args.lr)
    #optimizer = th.optim.RMSprop(model.parameters(), lr = args.lr)
    #optimizer = th.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True )
    lossbuf = [] 

    ## LOAD ALL NECESSARY INFORMATION
    # model weights
    if args.data =='cifar':
        model_layers_dc = load_cifar_1to1_layers_dc('cifar_base_kw')
    elif args.data == 'mnist':
        raise NotImplementedError
    else:
        raise NotImplementedError
    

    #bounds_indices = [0]
    #for layer_idx, layer in enumerate(model_layers_dc['fixed_layers']):
    #    if type(layer) is nn.Linear:
    #        bounds_indices.append(layer_idx+1)
    #    elif type(layer) is nn.Conv2d:
    #        bounds_indices.append(layer_idx+1)
    #    else:
    #        pass
    #bounds_indices.append(len(model_layers_dc['fixed_layers'])+1)
    #import pdb; pdb.set_trace()

    #####################################
    ## DEBUG
    files_train = files_train[:100]
    files_val = files_val[:10]
    #####################################

    graph_info_all = process_branches(files_train)

    graph_info_val = process_branches(files_val)

    # log of min, max  gradients of each layer
    #log_name = os.path.join(args.save_path, 'grad_log_{}_{}.txt'.format(args.exp_name, sub_exp))
    #loss_record = open(log_name,'w')    
    #keys_records = [i for i in model.state_dict().keys()]
    best_dev_acc = -1
    orig_accuracy = 0
    orig_loss = float("inf")
    current_plateau_epoch = 0
    tm_plateau_epoch = 0
    #lr_decay = (args.start_lr - args.end_lr)/(args.epoch//50-1)
    #lr_decay = 0.0001
    layers = {}
    layers['fixed_layers'] = [l.cuda() for l in model_layers_dc['fixed_layers']]


    for e in range(args.epoch):
        start_time = time.time()
        print('number of epoch: %d'%e)
        random.shuffle(graph_info_all)
        
        acc = 0
        graph_idx = 0
        while graph_idx < len(graph_info_all):
            if (graph_idx + args.batch_size <= len(graph_info_all)):
                graph_info_batch = graph_info_all[graph_idx: graph_idx+args.batch_size]
                graph_idx = graph_idx+args.batch_size
                # DEBUG
                #graph_info_batch = [graph_info_all[0], graph_info_all[1], graph_info_all[0]]
            else:
                graph_info_batch = graph_info_all[graph_idx:]
                graph_idx = len(graph_info_all)

            model.train(); optimizer.zero_grad()
        
            lower_bounds_temp = []
            upper_bounds_temp = []
            dual_vars_temp = []
            primals_temp = []
            layers['prop_layers'] = []
            masks = []
            primal_inputs = []
            rel_decisions = []

            for case in graph_info_batch:
                lower_bounds_temp.append(case['lower_bounds'])
                upper_bounds_temp.append(case['upper_bounds'])
                dual_vars_temp.append(case['dual_vars'])
                primals_temp.append(case['primals'])
                label = case['label']
                layers['prop_layers'].append(model_layers_dc[label].cuda())
                masks.append(case['rel_mask_1d'].cuda())
                primal_inputs.append(case['primal_input'].cuda())
                rel_dis_temp = case['rel_decision'][case['rel_mask_1d'].nonzero().view(-1)].cuda()
                #rel_dis_temp = case['rel_dec_01'].cuda()
                rel_decisions.append(rel_dis_temp)



            
            lower_bounds_temp = list(zip(*lower_bounds_temp))
            lower_bounds_all = [th.stack(i,0).cuda() for i in lower_bounds_temp]
            upper_bounds_temp = list(zip(*upper_bounds_temp))
            upper_bounds_all = [th.stack(i,0).cuda() for i in upper_bounds_temp]
            primals_temp = list(zip(*primals_temp))
            primals= [th.cat(i,0).cuda() for i in primals_temp]
            dual_vars_temp = list(zip(*dual_vars_temp))
            dual_vars = [th.cat(i,0).cuda() for i in dual_vars_temp]
            masks = th.stack(masks,0)
            primal_inputs = th.stack(primal_inputs, 1).squeeze(0)
            import pdb; pdb.set_trace()
            scores = model(lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_inputs, layers, masks)
            #import pdb; pdb.set_trace()
            # record accuracy of graph decision

            #scores_dom = scores.reshape(1,-1).cuda()
            rdises = []
            for i in range(len(scores)):
                _, choice = th.max(scores[i], 0)
                rdises.append(rel_decisions[i][choice])
            rdises = th.tensor(rdises)
            decision = (rdises>args.dec_threshold).float()
            bool_temp = decision.mean()
            acc += decision.sum()
            avgrdis = rdises.mean()
            
            rel_decisions = nn.utils.rnn.pad_sequence(rel_decisions, batch_first=True)
            
            if args.loss == 'mse':
                scores = [sigm(i) for i in scores]
                scores = nn.utils.rnn.pad_sequence(scores, batch_first=True)
                #scores = [(i-th.min(i))/(th.max(i)-th.min(i)) for i in scores]
                #import pdb; pdb.set_trace()
                loss = criterion(scores, rel_decisions).cuda()
                loss = loss/th.sum(masks)

            elif args.loss == 'hinge_rank':
                #hinge rank loss with threshold
                # first decide the label for each decision
                scores = nn.utils.rnn.pad_sequence(scores, batch_first=True)
                labels = th.zeros(rel_decisions.size())
                step_size = args.losstp
                steps = int(1/step_size)
                #thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
                for i in range(1, steps, 1):
                    labels[rel_decisions>i*step_size] += 1
                
                labels[rel_decisions==0.0] = -steps*10.0
                labels.cuda()
                loss = hinge_rank_loss(scores, labels, steps)

            else:
                raise NotImplementedError

            #pdb.set_trace()
            #loss = criterion(scores_dom, decision).cuda()
            #loss = criterion(sigm(scores_dom), decision).cuda()
            #loss = loss*weight
            #loss = th.mean(loss)
            if (args.logger ==True):
                xp.Loss_Train.update(loss)
                xp.Acc_Train.update(bool_temp)
                xp.Rdis_Train.update(avgrdis)
                    
            lossbuf.append(loss.item())
            loss.backward()

            optimizer.step()
            # pdb.set_trace() 
            # testing weights: if any weight becomes nan, record grad
            # and return
            check_nan = len([1 for i in model.parameters() if th.sum(th.isnan(i))!=0])
            if (check_nan != 0):
                loss_record.write('epoch number: {}\n'.format(e))
                loss_record.write('encountered nan\n')
                print('encountered nan\n')
                for j in range(len(keys_records)):
                    loss_record.write('grad of layer: {}'.format(keys_records[j]))
                    loss_record.write('\n')
                    loss_record.write('abs max: {}\n'.format(th.max(th.abs(grad_records[j]))))
                    loss_record.write('abs min: {}\n'.format(th.min(th.abs(grad_records[j]))))


                    print('grad of layer: {}'.format(keys_records[j]))
                    print('abs max: {}\n'.format(th.max(th.abs(grad_records[j]))))
                    print('abs min: {}\n'.format(th.min(th.abs(grad_records[j]))))
                loss_record.write('\n\n')
                loss_record.close()
                return
                    

            

            if graph_idx % (5*args.batch_size) == 0:
                temp = np.mean(lossbuf)
                #print(decision)
                #print(scores_dom)
                print(graph_idx, temp)
                lossbuf = []

        #record epoch loss


        #########################
        
        #after each epoch, save model 
        train_acc = acc/len(graph_info_all)
        print('epoch accuracy: ', train_acc)

        snapshot_prefix = os.path.join(args.save_path, 'snapshot_{}_{}_'.format(args.exp_name, sub_exp))
        snapshot_path = snapshot_prefix + 'acc_{:.3f}_loss_{:.4f}_epoch_{}.pt'.format(train_acc, loss.item(), e)
        th.save(model.state_dict(), snapshot_path)
        for f in glob.glob(snapshot_prefix + '*'):
            if (f != snapshot_path):
                os.remove(f)

        # validation
        model.eval()
        batch_size_val = 100
        with th.no_grad():
            acc_val = 0
            loss_val_buf =[]
            graph_idx_val = 0
            while graph_idx_val < len(graph_info_val):
                if (graph_idx_val + batch_size_val <= len(graph_info_val)):
                    graph_info_batch_val = graph_info_val[graph_idx_val: graph_idx_val + batch_size_val]
                    graph_idx_val = graph_idx_val + batch_size_val
                else:
                    graph_info_batch_val = graph_info_val[graph_idx_val:]
                    graph_idx_val = len(graph_info_val)

            
                lower_bounds_temp = []
                upper_bounds_temp = []
                dual_vars_temp = []
                primals_temp = []
                layers['prop_layers'] = []
                masks = []
                primal_inputs = []
                rel_decisions = []

                for case in graph_info_batch_val:
                    lower_bounds_temp.append(case['lower_bounds'])
                    upper_bounds_temp.append(case['upper_bounds'])
                    dual_vars_temp.append(case['dual_vars'])
                    primals_temp.append(case['primals'])
                    label = case['label']
                    layers['prop_layers'].append(model_layers_dc[label].cuda())
                    masks.append(case['rel_mask_1d'].cuda())
                    primal_inputs.append(case['primal_input'].cuda())
                    rel_dis_temp = case['rel_decision'][case['rel_mask_1d'].nonzero().view(-1)].cuda()
                    #rel_dis_temp = case['rel_dec_01'].cuda()
                    rel_decisions.append(rel_dis_temp)



                
                lower_bounds_temp = list(zip(*lower_bounds_temp))
                lower_bounds_all = [th.stack(i,0).cuda() for i in lower_bounds_temp]
                upper_bounds_temp = list(zip(*upper_bounds_temp))
                upper_bounds_all = [th.stack(i,0).cuda() for i in upper_bounds_temp]
                primals_temp = list(zip(*primals_temp))
                primals= [th.cat(i,0).cuda() for i in primals_temp]
                dual_vars_temp = list(zip(*dual_vars_temp))
                dual_vars = [th.cat(i,0).cuda() for i in dual_vars_temp]
                masks = th.stack(masks,0)
                primal_inputs = th.stack(primal_inputs, 1).squeeze(0)
                scores_val = model(lower_bounds_all, upper_bounds_all, dual_vars, primals, primal_inputs, layers, masks)
                # record accuracy of graph decision

                #scores_dom = scores.reshape(1,-1).cuda()
                rdises_val = []
                for i in range(len(scores_val)):
                    _, choice = th.max(scores_val[i], 0)
                    rdises_val.append(rel_decisions[i][choice])
                rdises_val = th.tensor(rdises_val)
                decision_val = (rdises_val>args.dec_threshold).float()
                bool_temp_val = decision_val.mean()
                acc_val += decision_val.sum()
                avgrdis_val = rdises_val.mean()
                
                rel_decisions = nn.utils.rnn.pad_sequence(rel_decisions, batch_first=True)
                if args.loss =='mse':
                    scores_val = [sigm(i) for i in scores_val]
                    #scores_val = [(i-th.min(i))/(th.max(i)-th.min(i)) for i in scores_val]
                    scores_val = nn.utils.rnn.pad_sequence(scores_val, batch_first=True)
                    #pdb.set_trace()
                    #loss = criterion(scores_dom, decision).cuda()
                    #loss = criterion(sigm(scores_dom), decision).cuda()
                    loss_val = criterion(scores_val, rel_decisions).cuda()
                    #loss = loss*weight
                    #loss = th.mean(loss)
                    loss_val = loss_val/th.sum(masks)
                    #loss = hinge_rank_loss(scores, decision)

                elif args.loss == 'hinge_rank':
                    #hinge rank loss with threshold
                    # first decide the label for each decision
                    scores_val = nn.utils.rnn.pad_sequence(scores_val, batch_first=True)
                    labels = th.zeros(rel_decisions.size())
                    step_size = args.losstp
                    steps = int(1/step_size)
                    #thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
                    for i in range(1, steps, 1):
                        labels[rel_decisions>i*step_size] += 1
                    
                    labels[rel_decisions==0.0] = -steps*10.0
                    labels.cuda()
                    loss_val = hinge_rank_loss(scores_val, labels, steps)

                else:
                    raise NotImplementedError

                loss_val_buf.append(loss_val.item())

                if (args.logger ==True):
                    xp.Loss_Val.update(loss_val)
                    xp.Acc_Val.update(bool_temp_val)
                    xp.Rdis_Val.update(avgrdis_val)

        #import pdb; pdb.set_trace()
        loss_val_avg = sum(loss_val_buf)/len(loss_val_buf)
        train_acc_val = acc_val/len(files_val)
        print('epoch validation accuracy: ', train_acc_val)
        print('epoch validation loss: ', loss_val_avg)

        #if train_acc_val < orig_accuracy:
        if loss_val_avg > orig_loss:
            current_plateau_epoch += 1
            tm_plateau_epoch +=1
        else:
            orig_loss =  loss_val_avg
            current_plateau_epoch = 0
            tm_plateau_epoch = 0

        if current_plateau_epoch == args.plateau_epoch:
            for g in optimizer.param_groups:
                current_lr = g['lr'] 
                new_lr = current_lr/5
                g['lr']= new_lr
                print(f'current epoch learning rate: {new_lr}')
                current_plateau_epoch = 0
        print('current plateau epoch number: ', current_plateau_epoch)
        print('current tm_plat epoch number: ', tm_plateau_epoch)

        if tm_plateau_epoch == args.tm_plateau_epoch:
            print('no improvement over ', tm_plateau_epoch, ' epochs: terminate')
            return

        #update best validation set
        if (train_acc_val > best_dev_acc):
            best_dev_acc = train_acc_val
            snapshot_prefix = os.path.join(args.save_path, 'best_snapshot_{}_{}_'.format(args.exp_name,sub_exp))
            snapshot_path = snapshot_prefix + 'val_acc_{:.3f}_loss_val_{:.4f}_epoch_{}.pt'.format(train_acc_val, loss_val.item(), e)
            th.save(model.state_dict(), snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

        if (args.logger ==True):
            xp.Loss_Train.log_and_reset()
            xp.Acc_Train.log_and_reset()
            xp.Rdis_Train.log_and_reset()
            xp.Loss_Val.log_and_reset()
            xp.Acc_Val.log_and_reset()
            xp.Rdis_Val.log_and_reset()
        



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', nargs = '?', default ='./cifar_kw_prox_m2_train_data/train', type= str, help = 'directory of training samples')
    parser.add_argument('--val', nargs = '?', default ='./cifar_kw_prox_m2_train_data/val', type= str, help = 'directory of  samples')
    parser.add_argument('--logger', action = 'store_true', help='turn logger on (default is off)')
    parser.add_argument('--T', nargs='?', default=2, type=int, help='number of embedding layer updates')
    parser.add_argument('--epoch', nargs ='?', default=200, type=int,  help='number of epoches')
    parser.add_argument('--batch_size', nargs ='?', default=5, type=int)
    parser.add_argument('--plateau_epoch', nargs ='?', default=10, type=int,  help='number of epoches')
    parser.add_argument('--tm_plateau_epoch', nargs ='?', default=20, type=int,  help='number of epoches')
    parser.add_argument('--p', nargs ='?', default=64, type=int,  help='dimension of embedding vectors')
    #parser.add_argument('--end_lr', nargs ='?', default=1e-4, type=float,  help='learning rate for adam')
    parser.add_argument('--start_lr', nargs ='?', default=1e-4, type=float,  help='learning rate for adam')
    parser.add_argument('--wd', nargs ='?', default=1e-4, type=float,  help='weight decay for adam')
    parser.add_argument('--dec_threshold', nargs ='?', default=0.9, type=float,  help='relative distance threshold that is used to binarize all relu decisions for a branch')
    #parser.add_argument('--save_path', nargs ='?', default = '/home/jodie/PLNN/PLNN-verification-private/graphnet/cifar_trained_models/m2', type=str,  help='the path to save all results')
    parser.add_argument('--save_path', nargs ='?', default = './gnn_branching/trained_models/cifar/', type=str,  help='the path to save all results')
    parser.add_argument('--exp_name', type=str,  help='name of the experiment for the logger')
    parser.add_argument('--loss', type=str, default='hinge_rank', help='type of loss function')
    parser.add_argument('--data', type=str, default='cifar',  help='dataset')
    parser.add_argument('--losstp', type=float, default=0.1, help='type of loss function')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    th.cuda.manual_seed_all(1)
    args = get_args()
    if args.exp_name is None:
        exp_name = f'BatchedCIFARm2_Adam_{args.loss}_losstp_{args.losstp}_{args.start_lr}_wd_{args.wd}_epoch_{args.epoch}_p_{args.p}_T_{args.T}_batch_{args.batch_size}'
        print(exp_name)
    else:
        exp_name = args.exp_name
    args.save_path = os.path.join(args.save_path, exp_name)
    if (os.path.exists(args.save_path) == False):
        os.makedirs(args.save_path)
    #cross_validation
    for i in range(1):
    #    idxes = [1,2,3,4]
    #    val_name = 'train'+f'{idxes.pop(i)}'
    #    files_val = glob.glob(args.train + val_name +'/*')
    #    files_train = []
    #    for j in idxes: files_train.extend(glob.glob(args.train+f'train{j}'+'/*'))
        files_train = glob.glob(args.train+'/*')
        files_val = glob.glob(args.val+'/*')
    
#----------------------------------
# prepare logging
#----------------------------------

# create experiment
        if args.logger:
            xp = logger.Experiment('{}'.format(exp_name), use_visdom = True, 
                    visdom_opts={'server': 'http://atlas.robots.ox.ac.uk', 'port': 9101}, time_indexing=False, xlabel='Epoch')

            print("hello")
            xp.log_config({'exp': exp_name, 'start_lr':args.start_lr,  'n_epochs': args.epoch, 'embedding_dim': args.p, 'passes': args.T, 'batch_size': args.batch_size})
            xp.AvgMetric(name = 'loss', tag = 'Train')
            xp.AvgMetric(name = 'loss', tag = 'Val')
            xp.AvgMetric(name = 'acc', tag = 'Train')
            xp.AvgMetric(name = 'acc', tag = 'Val')
            xp.AvgMetric(name = 'rdis', tag = 'Train')
            xp.AvgMetric(name = 'rdis', tag = 'Val')

        train(files_train, files_val, args, i)

