import numpy as np
import pickle
import os 

path = 'resources/mnist/evaluation_images'
if not os.path.exists(path):
    os.makedirs(path)

with open('../benchmark/mnist/oval/mnist_images/labels','r') as f:
    labels = f.read().split(",")
labels = [int(l) for l in labels if len(l)>0]
f.close()

for i in range(25):
    with open(f'../benchmark/mnist/oval/mnist_images/image{i+1}','r') as f:
        im = f.read().split(",")
    im = [np.float32(k) for k in im if len(k)>0]
    im = np.array(im,dtype='float32') / 255.0
    with open(f'{path}/image{i+1}.pkl', 'wb') as f:
        pickle.dump( (im,labels[i]), f)

