function [rb , vt]= verifyMnist(netfile,epsilon,i,reachMethod,relaxFactor)
%% load network and input set
load(netfile, 'net');

%% verify the image
[IS, Labels] = getInputSetMnist(epsilon);

[~, rb, cE, cands, vt] = net.evaluateRBN(IS(i), Labels(i), reachMethod, 1, relaxFactor);

%% save counter Examples
filename = 'results/nist_0.'+string(epsilon*10)+'_CE.mat';
if isfile(filename)
    load(filename, 'cE1', 'cands1');
end

 cE1{i} = cE;
 cands1{i} = cands;

save(filename,'cE1','cands1');

end



function [IS, Lbls] = getInputSetMnist(eps)
%% construct input set with specific eps
if eps==0.1
    load IS_01.mat IS_01 Labels;
    IS = IS_01;
    Lbls = Labels;
elseif eps==0.3
    load IS_03.mat IS_03 Labels;
    IS = IS_03;
    Lbls = Labels;
end
end