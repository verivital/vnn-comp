%% Load images and construct imagestar input set

T = csvread("cifar10_test.csv");
%load CIFAR10.mat;
im_data = T(:,2:3073);
im_labels = T(:,1);

N = size(T, 1);% number of imagestar input sets (100 imagestars)

IS = ImageStar;
Labels = im_labels + 1; % labels corresponding to the disturbanced images
load Norm.mat;

for i=1:N
    fprintf("\nConstructing %d^th ImageStar input set", i);
    im = im_data(i,:);
    im = im';
    im = im/255;
    epsilon = 2;
    eps = epsilon/255; 
    ub = im + eps;
    lb = im - eps;
    ub(ub>1) = 1;
    lb(lb<0) = 0;
    %norm_im = normalize(mean_data,std_data,im);
    lb = normalize(mean_data,std_data,lb);
    ub = normalize(mean_data,std_data,ub);
    filename='ImagestarSets/IS_'+string(epsilon)+'_255_'+string(i)+'.mat';
    IS = ImageStar(lb, ub);
    save(filename,'IS');
    
    epsilon=8;
    eps = epsilon/255; 
    ub = im + eps;
    lb = im - eps;
    ub(ub>1) = 1;
    lb(lb<0) = 0;
    %norm_im = normalize(mean_data,std_data,im);
    lb = normalize(mean_data,std_data,lb);
    ub = normalize(mean_data,std_data,ub);
    IS = ImageStar(lb, ub);
    filename='ImagestarSets/IS_'+string(epsilon)+'_255_'+string(i)+'.mat';
    save(filename,'IS');
end

function normalized_image = normalize(mean_data,std_data,Image)
if ~all(mean_data==0) && ~all(std_data==1)
    I=reshape(Image,3,1024);
    for i=1:3
        J(:,:,i)=reshape(I(i,:),32,32);
        J(:,:,i)=J(:,:,i)';
    end
    for i=1:3
      norm_im(:,:,i) = (J(:,:,i) - mean_data(i))/std_data(i);
    end
    normalized_image=norm_im;
else 
    normalized_image = Image-0.5;
end

end