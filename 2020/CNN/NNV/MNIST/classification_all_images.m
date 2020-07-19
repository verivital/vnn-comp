Image=csvread('mnist_test.csv');
Labels=Image(:,1);
Labels=Labels+1;
Image=Image(:,2:785);
Image=Image/255;
mean_data = 0.1307;
std_data = 0.3018;
Image=(Image-mean_data)/std_data;

for i=1:100
im=Image(i,:);
im=reshape(im,28,28);
Y=net.evaluate(im');
[~,pred(i)]=max(Y);
end
sum(pred'==Labels)