The provided models have different network architectures and are of different sizes. Specifically, there is a base model with 2 convolutional layers followed by 2 fully connected layers. The other two are larger models: a wide model that has the same number of layers as the base model but more hidden units in each layer and a deep model that has more layers but a similar number of hidden units in each convolutional layer to the base model. All three models are trained robustly using the method provided by Kolter and Wong [1]. All three models are available in both PyTorch and ONNX format.

Verification properties:
The following verification properties are considered: given an image x for which the model correctly predicts the label y, verify that the trained network will not make a mistake by labelling a slightly perturbed image as y’ (y’ \neq y). The label y’ is randomly chosen at the beginning and the allowed perturbation is determined by an epsilon value under infinity norm. The epsilon values are found via binary searches with predefined timeouts.
Overall, on the base model, more than 1500 verification properties are collected with three difficulty levels and a one hour timeout. Also 300 properties are collected for the wide model and 250 properties for the deep model using 2-hour timeouts. 


If there are 1500 properties with a one hour timeout and 550 properties with a 2 hour timeout. There are 2 smaller subsets for each of the three models; one with 20, and another with 100 properties containing the first subset (https://github.com/oval-group/GNN_branching). Using a timeout of 1 hour for all properties, should suffice. As the majority of properties can be verified fairly quickly and because one can solve different properties in parallel, it should be a lot quicker to run experiments than the theoretical upper bounds you mentioned.

The pandas tables now have three columns only: all images are taken from the cifar20 test set and the Idx column refers to the image index; the Eps value defines the epsilon-sized l_infinity ball around the image; and finally the prop value defines the property we are verifying against, as we are doing 1-vs-1 verification. All properties are UNSAT, meaning that the network is robust around each image


The models and the pandas tables with all verification properties can be found at https://github.com/oval-group/GNN_branching/tree/master/onnx_models and https://github.com/oval-group/GNN_branching/tree/master/cifar_exp respectively. These verification datasets have already been used in two published works [2,3].

[1] Eric Wong and Zico Kolter. Provable defenses against adversarial examples via the convex outer adversarial polytope. International Conference on Machine Learning, 2018.

[2] Jingyue Lu and M. Pawan Kumar. Neural Network Branching for Neural Network Verification. International Conference on Learning Representations, 2020.

[3] Rudy Bunel and Alessandro De Palma and Alban Desmaison and Krishnamurthy Dvijotham and Pushmeet Kohli and Philip H. S. Torr and M. Pawan Kumar. Lagrangian Decomposition for Neural Network Verification.  Uncertainty in Artificial Intelligence, 2020.
