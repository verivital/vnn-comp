function [rb , vt]= verifyCifar(netfile,epsilon,i,reachMethod,relaxFactor)
which linprog
%% load network and input set
load(netfile,'net');
load('Labels.mat','Labels');

%% verify the image

inputSetFile='ImagestarSets/IS_'+string(epsilon)+'_255_'+string(i)+'.mat';%
load(inputSetFile, 'IS');

[~, rb, cE, cands, vt] = net.evaluateRBN(IS, Labels(i), reachMethod, 1, relaxFactor);

%% save counter examples
filename = 'results/cifar_'+string(epsilon)+'_CE.mat';

if isfile(filename)
    load(filename,'cE1','cands1');
end

cE1{i} = cE;
cands1{i} = cands;

save(filename,'cE1','cands1');

end


function IS = getInputSetCifar(eps,imNum)
% construct input set with specific eps
filename='IS_'+string(eps)+'_255_'+string(imNum)+'.mat';
load(filename, 'IS');
end