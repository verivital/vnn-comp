%% Load images, labels and construct imagestar input set
N = 16;
Labels = dlmread('./images/labels');
Labels = Labels(1:N)+1;% labels corresponding to the disturbanced images


S_eps_5(N) = ImageStar;
S_eps_12(N) = ImageStar;

for i=1:N
    fprintf("\nConstructing %d^th ImageStar input set", i);
    file='./images/image'+string(i);
    im=dlmread(file);
    im=im(1:784);
    
    eps = 5; 
    ub = im + eps;
    lb = im - eps;
    ub(ub>255) = 255;
    lb(lb<0) = 0;
    S_eps_5(i) = ImageStar(lb, ub);
    
    eps = 12; 
    ub = im + eps;
    lb = im - eps;
    ub(ub>255) = 255;
    lb(lb<0) = 0;
    S_eps_12(i) = ImageStar(lb, ub);

end

save inputStarSets.mat S_eps_12 S_eps_5 Labels;