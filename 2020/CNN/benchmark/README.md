This folder contains 2 MNIST and 2 CIFAR10 Convolutional networks in the ONNX format from:
https://github.com/eth-sri/colt/tree/master/trained_models/onnx

The epsilon values of the L_oo ball for the MNIST networks are 0.1 and 0.3 while those for the CIFAR10 networks
are 2/255 and 8/255. The network names contain epsilon values. All networks expect input images to be first 
normalized to be in the range [0,1]. Next, standard mean and deviations should be applied to the normalized 
images before passing them to the networks. The means and standard deviations for MNIST and CIFAR10 can be found here:
https://github.com/eth-sri/colt/blob/acea4093d0eebf84b49a7dd68ca9a28a08f86e67/code/loaders.py#L7

The first 100 images of the MNIST and CIFAR10 test set should be fine so that there is a good mix of easy, medium,
and hard instances for verification. They can be found here:
https://github.com/eth-sri/eran/tree/master/data
Images that are not correctly classified can be filtered out. 

The number of correctly classified images for the networks are as follows:
1. mnist_0.1.onnx: 100/100
2. mnist_0.3.onnx: 99/100
3. cifar10_2_255.onnx: 82/100
4. cifar10_8_255.onnx: 56/100

Steps:
1. Check if the classification is correct and discard the incorrectly classified images.
2. With the correctly classified images, check robustness with a an l_infinity norm with epsilon value mentioned in the file name
3. Report safe, unsafe, or timeout(between 1-5 minutes depending on the size of the network) for each network.
