#!/bin/bash

st_ratio=0.7
depth_power=20.0
splitters=1
workers=4
offline_dep=False
online_dep=False
ideal_cuts=False
timeout=21600
logfile="evaluation/acas_hard.log"


python3 evaluation/tools/prepare_acas_nets.py


echo model,property,result,time > ${logfile}

python3 . --property acas --acas_prop 0 --net resources/acas/models/acas_4_6.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile

python3 . --property acas --acas_prop 0 --net resources/acas/models/acas_4_8.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile

python3 . --property acas --acas_prop 1 --net resources/acas/models/acas_3_3.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile

python3 . --property acas --acas_prop 1 --net resources/acas/models/acas_4_2.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile

python3 . --property acas --acas_prop 1 --net resources/acas/models/acas_4_9.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile

python3 . --property acas --acas_prop 1 --net resources/acas/models/acas_5_3.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile

python3 . --property acas --acas_prop 2 --net resources/acas/models/acas_3_6.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile

python3 . --property acas --acas_prop 2 --net resources/acas/models/acas_5_1.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile

python3 . --property acas --acas_prop 7 --net resources/acas/models/acas_1_9.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile

python3 . --property acas --acas_prop 9 --net resources/acas/models/acas_3_3.h5 --st_ratio $st_ratio --depth_power $depth_power --splitters $splitters --workers $workers --offline_dep $offline_dep --online_dep $online_dep --ideal_cuts $ideal_cuts --timeout $timeout --logfile $logfile
