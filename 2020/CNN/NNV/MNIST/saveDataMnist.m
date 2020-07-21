function saveDataMnist(epsilon,rb,vt)
%% print the result
verify_time = sum(vt);
safe = sum(rb==1);
unsafe = sum(rb==0);
unknown = sum(rb==2);
timeOut = sum(rb==3);
incorrectlyClassified = sum(rb==4);
T = table(epsilon, safe, unsafe, unknown, timeOut,incorrectlyClassified, verify_time);

filename = 'results/mnist_0.'+string(epsilon*10)+'_results.mat';
save(filename,'T', 'rb', 'vt');
end