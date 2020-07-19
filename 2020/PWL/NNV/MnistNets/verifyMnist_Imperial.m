function [rb,vt] = verifyMnist_Imperial(netfile, epsilon, i, reachMethod, relaxFactor,numLayer)
%% load network and input set
load(netfile,'net');

%% verify the image
[IS, labels] = getInputSetMnistImp(epsilon);
[~, rb, cE, cands, vt] = net.evaluateRBN(IS(i), labels(i), reachMethod,1,relaxFactor);

%% save counter Examples
filename = 'results/mnist_'+string(numLayer)+'L_0.0'+string(epsilon*100)+'_CE.mat';
if isfile(filename)
    load(filename, 'cE1', 'cands1');
end

 cE1{i} = cE;
 cands1{i} = cands;

save(filename,'cE1','cands1');


end

function [IS, Lbls] = getInputSetMnistImp(eps)
% construct input set with specific eps

if eps== 0.02
    load inputStarSets.mat S_eps_002 Labels;
    IS = S_eps_002;
    Lbls = Labels;
elseif eps== 0.05
    load inputStarSets.mat S_eps_005 Labels;
    IS = S_eps_005;
    Lbls = Labels;
end
end