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


