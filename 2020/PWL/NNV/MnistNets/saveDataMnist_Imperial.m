function saveDataSigm(epsilon,rb1,vt1,numLayer)
%% buid table 
verify_time = sum(vt1);
safe = sum(rb1==1);
unsafe = sum(rb1 == 0);
unknown = sum(rb1 == 2);
timeOut = sum(rb1 ==3);

T = table(epsilon, safe, unsafe, unknown,timeOut, verify_time)

filename = 'results/mnist_'+string(numLayer)+'L_0.0'+string(epsilon*100)+'_results.mat';
save(filename, 'T','rb1', 'vt1');
end