function [rb,vt] = verifySigNet(netfile, epsilon, i, reachMethod, relaxFactor,numLayer)
%% load network and input set
load(netfile,'net');
%% verify the image
[IS, labels] = getInputSetMnist(epsilon);
[~, rb, cE, cands, vt] = net.evaluateRBN(IS(i), labels(i), reachMethod,1,relaxFactor);

%% save counter Examples
filename = 'results/sig_'+string(numLayer)+'L_'+string(epsilon)+'_CE.mat';
if isfile(filename)
    load(filename, 'cE1', 'cands1');
end

 cE1{i} = cE;
 cands1{i} = cands;

save(filename,'cE1','cands1');

end

function [IS, Lbls] = getInputSetMnist(eps)
% construct input set with specific eps

if eps== 5
    load inputStarSets.mat S_eps_5 Labels;
    IS = S_eps_5;
    Lbls = Labels;
elseif eps== 12
    load inputStarSets.mat S_eps_12 Labels;
    IS = S_eps_12;
    Lbls = Labels;
end
end
