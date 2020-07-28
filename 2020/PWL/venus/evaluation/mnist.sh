#!/bin/bash
st_ratio=0.4
depth_power=4
splitters=1
workers=2
offline_dep=True
online_dep=True
ideal_cuts=True
timeout=900
logfile="evaluation/mnist.log"

python3 evaluation/tools/prepare_mnist_imgs.py

echo model,image,radius,result,time >> ${logfile}
for model in mnist-net_256x2.onnx mnist-net_256x4.onnx mnist-net_256x6.onnx
do
	for radius in 0.02 0.05
	do
		for i in {1..25}
		do
			python3 . --property lrob --lrob_input resources/mnist/evaluation_images/image${i}.pkl --lrob_radius $radius --net ../benchmark/mnist/oval/${model}  --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile
		done
	done
done

