Piece-Wise Linear Neural Network:
-----------------------------

This folder contains the codes and the results for the 'PWL' sub-category of **VNN-COMP 2020**. 

ACAS Xu and MNISTNets were the networks focused for this sub-category.

General File Structure:
-----------------------
1. The 'results' folder contains all the results for the networks in respective dataset specific(i.e MNIST or AcasXu) folders.
2. 'executeAcasXu.py' and 'executeImperial.py' are the main execution files which check the verification results with a threshold time out.
3. 'run_PWL.sh' is the top level script for running all the tests in this sub-category.

NNV Installation:
-----------------
For the installation of NNV, one can follow thw directions of the link: https://github.com/verivital/nnv/blob/master/README.md
The installation needs to be done once, before performing any of the NNV-tests.

P.S. NNV used gurobi linprog optimization ('glpk' can also be used by mantioning it in the argument of verifyRBN function) and for that gurobi 
installation and licensing is required prior to the testing. For setting gurobi linprog to MATLAB path, this link will be helpful:

https://www.gurobi.com/documentation/9.0/quickstart_mac/matlab_setting_up_grb_for_.html

Execution Process:
------------------
1. `cd` to the MATLAB installation directory.
2. Make sure the script is executable by running `chmod +x <path to run_PWL.sh>/run_PWL.sh`from terminal.
3. Run `<path to run_PWL.sh>/run_PWL.sh`.
4. For the first time, run the following matlab command in the terminal to add the nnv paths:
    `matlab -nodisplay -r "addpath(genpath(<path to 'startup_nnv' file>)); addpath(genpath(<path to 'code' folder inside NNV>)); startup_nnv; savepath;"`
5. The run script also lists the configurable parameters for running the test cases.


For any further queries regarding running the tests, please contact: Neelanjana Pal(neelanjana.pal@vanderbilt.edu)
