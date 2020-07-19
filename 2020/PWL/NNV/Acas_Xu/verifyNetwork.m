%% verification of all the networks for a particular property
function [safe, vt] = verifyNetwork(n1, n2, property_id, method, numCores)
    [F, I] = constructNetwork_inputSet(n1, n2, property_id);
    [u_mat, u_vec, numUVec] = getSpecs(property_id);
    %fprintf('size of u_mat [%d %d] and u_vec [%d %d]',size(u_mat,1), size(u_mat,2),size(u_vec,1),size(u_vec,2));
    if size(I,2) ==1
        [safe, vt] = robustness_verify(F,I,u_mat,u_vec,method,numCores,numUVec);
    elseif size(I,2) ==2
        [safe1, vt1] = robustness_verify(F,I(1),u_mat,u_vec,method,numCores,numUVec);
        [safe2, vt2] = robustness_verify(F,I(2),u_mat,u_vec,method,numCores,numUVec);
        vt=vt1+vt2;
        if safe1 == 0 || safe2 == 0
            safe = 0;
        elseif safe1 == 1 && safe2 == 1
            safe = 1;
        else
            safe = 2;
        end
    end
    %[safe, vt, ~] = F.verify(I, U, method, numCores);
end

%% check robustness of a network w.r.t a given Starset
function [safe,t] = robustness_verify(net,S,U,u_vec,method, numCores,numUVec)
    safe = 2;
    [R,t]=net.reach(S,method,numCores);
    
    if numUVec>1 
       l = size(U,1)/numUVec;
       pos = 1;
       for i=1:numUVec
           for j = 1:length(R)
                if isempty(R(:,j).intersectHalfSpace(U(pos:pos+l-1,:), u_vec))
                      safe = 1;
                else
                    if method == string('approx-star')
                          safe = 2; 
                          break
                    else
                         safe = 0;
                         break
                    end
                end
           end
           pos = pos + l;
        end
     else
        for j = 1:length(R)
              if isempty(R(j).intersectHalfSpace(U, u_vec))
                  safe = 1;
              else
                  if method == string('approx-star')
                      safe = 2; 
                      break
                else
                     safe = 0;
                     break
                 end
              end
         end
     end
    %t= t+toc(vt)-vt;
    %fprintf('Time:  %d',t);
end

%% construct a network from network id
function [F, I] = constructNetwork_inputSet(n1,n2,property_id)    
    addpath(genpath("/home/neelanjana/Documents/MATLAB/nnv/code/nnv/examples/NN/VNN_COMP2020/FM2019_Journal/ACAS_Xu/nnet-mat-files/"));
    load(['ACASXU_run2a_',num2str(n1),'_',num2str(n2),'_batch_2000.mat']); 
    Layers = [];
    n = length(b);
    for i=1:n - 1
        bi = cell2mat(b(i));
        Wi = cell2mat(W(i));
        Li = LayerS(Wi, bi, 'poslin');
        Layers = [Layers Li];
    end
    bn = cell2mat(b(n));
    Wn = cell2mat(W(n));
    Ln = LayerS(Wn, bn, 'purelin');
    Layers = [Layers Ln];
    F = FFNNS(Layers);
    
    [lb, ub] = getInputs(property_id);
    % normalize input
    for i=1:5
        lb(i,:) = (lb(i,:) - means_for_scaling(i))/range_for_scaling(i);
        ub(i,:) = (ub(i,:) - means_for_scaling(i))/range_for_scaling(i);   
    end
    if size(lb,2)==1
        I = Star(lb, ub);
    else
        for i=1:size(lb,2)
            I(i)=Star(lb(:,2),ub(:,2));
        end
    end
        
end

%% get input sets
function [lb, ub] = getInputs(property_id)
    if property_id == 1 || property_id == 2
        % Input Constraints
        % 55947.69 <= i1(\rho) <= 60760,
        % -3.14 <= i2 (\theta) <= 3.14,
        %-3.14 <= i3 (\shi) <= -3.14
        % 1145 <= i4 (\v_own) <= 1200, 
        % 0 <= i5 (\v_in) <= 60
        lb = [55947.691; -3.141592; -3.141592; 1145; 0];
        ub = [60760; 3.141592; 3.141592; 1200; 60];

    elseif property_id == 3
        % Input Constraints
        % 1500 <= i1(\rho) <= 1800,
        % -0.06 <= i2 (\theta) <= 0.06,
        % 3.1 <= i3 (\shi) <= 3.14
        % 980 <= i4 (\v_own) <= 1200, 
        % 960 <= i5 (\v_in) <= 1200
        % ****NOTE There was a slight mismatch of the ranges of
        % this i5 input for the conference paper, FM2019 "Star-based Reachability of DNNs"
        lb = [1500; -0.06; 3.1; 980; 960];
        ub = [1800; 0.06; 3.14; 1200; 1200];
    elseif property_id == 4        
        % Input Constraints
        % 1500 <= i1(\rho) <= 1800,
        % -0.06 <= i2 (\theta) <= 0.06,
        % (\shi) = 0
        % 1000 <= i4 (\v_own) <= 1200, 
        % 700 <= i5 (\v_in) <= 800
        lb = [1500; -0.06; 0; 1000; 700];
        ub = [1800; 0.06; 0; 1200; 800];
        
    elseif property_id == 5        
        % Input Constraints
        % 250 <= i1(\rho) <= 400,
        % 0.26 <= i2 (\theta) <= 0.4,
        % -3.141592 <= (\shi) <= -3.141592 + 0.005,
        % 100 <= i4 (\v_own) <= 400, 
        % 0 <= i5 (\v_in) <= 400
        lb = [250; 0.26; -3.141592; 100; 0];
        ub = [400; 0.4; -3.141592+0.005; 400; 400];
        
    elseif property_id == 6        
        % Input Constraints
        % 12000 <= i1(\rho) <= 62000,
        % 0.07 <= i2 (\theta) <= 3.141592, V -3.141592 <= (\theta) <= -0.7
        % -3.141592 <= (\shi) <= -3.141592 + 0.005,
        % 100 <= i4 (\v_own) <= 1200, 
        % 0 <= i5 (\v_in) <= 1200
        lb1 = [12000; 0.7; -3.141592; 100; 0];
        ub1 = [62000; 3.141592; -3.141592+0.005; 1200; 1200];
        
        lb2 = [12000; -3.141592; -3.141592; 100; 0];
        ub2 = [62000; -0.7; -3.141592+0.005; 1200; 1200];
        
        lb = [lb1, lb2];
        ub = [ub1, ub2];
        
    elseif property_id == 7        
        % Input Constraints
        % 0 <= i1(\rho) <= 60760,
        % -3.141592.06 <= i2 (\theta) <= 3.141592,
        % -3.141592 <= (\shi) <= 3.141592
        % 100 <= i4 (\v_own) <= 1200, 
        % 0 <= i5 (\v_in) <= 800
        lb = [0; -3.141592; -3.141592; 100; 0];
        ub = [60760; 3.141592; -3.141592; 1200; 800];
    elseif property_id == 8        
        % Input Constraints
        % 0 <= i1(\rho) <= 60760,
        % -3.141592.06 <= i2 (\theta) <= 3.141592,
        % -0.1 <= (\shi) <= 0.1
        % 600 <= i4 (\v_own) <= 1200, 
        % 600 <= i5 (\v_in) <= 1200
        lb = [0; -3.141592; -0.1; 600; 600];
        ub = [60760; -0.75*3.141592; 0.1; 1200; 1200];
    elseif property_id == 9        
        % Input Constraints
        % 2000 <= i1(\rho) <= 7000,
        % -0.4 <= i2 (\theta) <= -0.14,
        % -3.141592 <= (\shi) <= -3.141592+0.01
        % 100 <= i4 (\v_own) <= 150, 
        % 0 <= i5 (\v_in) <= 150
        lb = [2000; -0.4; -3.141592; 100; 0];
        ub = [7000; -0.14; -3.141592 + 0.01; 150; 150];
    elseif property_id == 10        
        % Input Constraints
        % 36000 <= i1(\rho) <= 60760
        % 0.7 <= i2 (\theta) <= 3.141592,
        % -3.141592 <= (\shi) <= -3.141592+0.01
        % 900 <= i4 (\v_own) <= 1200, 
        % 600 <= i5 (\v_in) <= 1200
        lb = [36000; 0.7; -3.141592; 900; 600];
        ub = [60760; -3.141592; -3.141592 + 0.01; 1200; 1200];
    
    else
        error('Invalid property ID');
    end
end

%% unsafe region
function [unsafe_mat, unsafe_vec, numUVec] = getSpecs(property_id)

    if property_id == 1 
        % output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        % verify safety: COC <= 1500 or x1 <= 1500 after normalization
        unsafe_mat = [-1 0 0 0 0];
        % safe region before normalization
        % x1' <= (1500 - 7.5189)/373.9499 = 3.9911
        unsafe_vec=-3.9911;
        %U = HalfSpace([-1 0 0 0 0], -3.9911); % unsafe region x1' > 3.9911
        numUVec = 1;
    elseif property_id == 2
        
        % output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        % safety property: COC is not the maximal score
        % unsafe region: COC is the maximal score: x1 >= x2; x1 >= x3; x1 >= x4, x1
        % >= x5
        unsafe_mat = [-1 1 0 0 0; -1 0 1 0 0; -1 0 0 1 0; -1 0 0 0 1];
        unsafe_vec = [0; 0; 0; 0];
        %U = HalfSpace(unsafe_mat, unsafe_vec);
        numUVec = 1;
    elseif property_id == 3 || property_id == 4
        
        % output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        % safety property: COC is not the minimal score
        % unsafe region: COC is the minimal score: x1 <= x2; x1 <= x3; x1 <= x4, x1
        % <= x5
        unsafe_mat = [1 -1 0 0 0; 1 0 -1 0 0; 1 0 0 -1 0; 1 0 0 0 -1];
        unsafe_vec = [0; 0; 0; 0];
        numUVec = 1;
    elseif property_id == 5
        
        % output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        % safety property: Strong Right is the minimal score
        % unsafe region: Strong Right is not the minimal score: x5 >= x2 OR
        % x5 >= x3 OR x5 >= x4 OR x5 >= x1
        unsafe_mat1 = [1 0 0 0 -1];
        unsafe_mat2 = [0 1 0 0 -1];
        unsafe_mat3 = [0 0 1 0 -1];
        unsafe_mat4 = [0 0 0 1 -1];
        unsafe_vec = 0;
        unsafe_mat = [unsafe_mat1; unsafe_mat2; unsafe_mat3; unsafe_mat4];
        numUVec = 4;
    elseif property_id == 6 || property_id == 10
        
        % output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        % safety property: COC is the minimal score
        % unsafe region: COC is not the minimal score: x1 >= x2 OR x1 >= x3 OR x1 >= x4 OR x1
        % >= x5
        unsafe_mat1 = [-1 1 0 0 0];
        unsafe_mat2 = [-1 0 1 0 0];
        unsafe_mat3 = [-1 0 0 1 0];
        unsafe_mat4 = [-1 0 0 0 1];
        unsafe_vec = 0;
        unsafe_mat = [unsafe_mat1; unsafe_mat2; unsafe_mat3; unsafe_mat4];
        numUVec = 4;
    elseif property_id == 7
        
        % output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        % safety property: Strong Left and Strong Right are never the
        % minimal scores
        % unsafe region: Strong Left and Strong Right are always the minimal score: 
        % x4 <= x2; x4 <= x3; x4 <= x1, x4<= x5  OR
        % x5 <= x2; x5 <= x3; x5 <= x4, x5<= x1  
        unsafe_mat1 = [1 0 0 0 -1; 0 1 0 0 -1; 0 0 1 0 -1; 0 0 0 1 -1];
        unsafe_mat2 = [1 0 0 -1 0; 0 1 0 -1 0; 0 0 1 -1 0; 0 0 0 -1 1];
        unsafe_vec = [0; 0; 0; 0];
        unsafe_mat = [unsafe_mat1; unsafe_mat2];
        numUVec = 2;
    elseif property_id == 8
        
        % output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        % safety property: Weak Left is minimal or COC is minimal
        % unsafe region: Neither of Weak Left or COC is minimal:
        % (x1 >= x2 or x1 >= x3 or x1 >= x4 or x1 >= x5) AND (x2 >= x1 or x2 >= x3 or x2 >= x4 or 
        % x2 >= x5)
        
        unsafe_mat1 = [-1 1 0 0 0; 0 -1 1 0 0];
        unsafe_mat2 = [-1 1 0 0 0; 0 -1 0 1 0];
        unsafe_mat3 = [-1 1 0 0 0; 0 -1 0 0 1];
        unsafe_mat4 = [-1 0 1 0 0; 1 -1 0 0 0];
        unsafe_mat5 = [-1 0 1 0 0; 0 -1 1 0 0];
        unsafe_mat6 = [-1 0 1 0 0; 0 -1 0 1 0];
        unsafe_mat7 = [-1 0 1 0 0; 0 -1 0 0 1];
        unsafe_mat8 = [-1 0 0 1 0; 1 -1 0 0 0];
        unsafe_mat9 = [-1 0 0 1 0; 0 -1 1 0 0];
        unsafe_mat10 = [-1 0 0 1 0; 0 -1 0 1 0];
        unsafe_mat11 = [-1 0 0 1 0; 1 0 0 0 -1];
        unsafe_mat12 = [-1 0 0 0 1; 1 -1 0 0 0];
        unsafe_mat13 = [-1 0 0 0 1; 0 -1 1 0 0];
        unsafe_mat14 = [-1 0 0 0 1; 0 -1 0 1 0];
        unsafe_mat15 = [-1 0 0 0 1; 0 -1 0 0 1];
        unsafe_vec = [0; 0];
        unsafe_mat = [unsafe_mat1; unsafe_mat2;unsafe_mat3;unsafe_mat4;...
            unsafe_mat5;unsafe_mat6;unsafe_mat7;unsafe_mat8;unsafe_mat9;...
            unsafe_mat10;unsafe_mat11;unsafe_mat12;unsafe_mat13;unsafe_mat14;unsafe_mat15];
        numUVec = 15;
    elseif property_id == 9
        
        % output: [x1 = COC; x2 = Weak Left; x3 = Weak Right; x4 = Strong Left; x5 = Strong Right]
        % safety property: Strong Left is the minimal score
        % unsafe region: Strong Left is never the minimal score: x4 <= x2 OR x4 <= x3 OR x4 <= x1 OR 
        % x4<= x5
        unsafe_mat1 = [1 0 0 -1 0];
        unsafe_mat2 = [0 1 0 -1 0];
        unsafe_mat3 = [0 0 1 -1 0];
        unsafe_mat4 = [0 0 0 -1 1];
        unsafe_vec = 0;
        unsafe_mat = [unsafe_mat1; unsafe_mat2; unsafe_mat3; unsafe_mat4];
        numUVec = 4;
    else
        error('Invalid property ID');
    end
end