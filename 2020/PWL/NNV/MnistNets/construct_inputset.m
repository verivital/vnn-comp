%% Load images, labels and construct imagestar input set
N = 25;
Labels = dlmread('./mnist_images/labels');
Labels = Labels(1:N)+1;% labels corresponding to the disturbanced images


S_eps_5(N) = ImageStar;
S_eps_12(N) = ImageStar;

for i=1:N
    fprintf("\nConstructing %d^th ImageStar input set", i);
    file='./mnist_images/image'+string(i);
    im=dlmread(file);
    im=im(1:784)/255;
    
    eps = 0.02; 
    ub = im + eps;
    lb = im - eps;
    ub(ub>1) = 1;
    lb(lb<0) = 0;
    S_eps_002(i) = ImageStar(lb, ub);
    
    eps = 0.05; 
    ub = im + eps;
    lb = im - eps;
    ub(ub>1) = 1;
    lb(lb<0) = 0;
    S_eps_005(i) = ImageStar(lb, ub);

end

save inputStarSets.mat S_eps_002 S_eps_005 Labels;