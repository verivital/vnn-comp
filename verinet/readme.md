# VNN-COMP 2020 Instructions

1) Run pipenv install from ".../vnn-comp/verinet/".
2) Download the VeriNet source code from: https://vas.doc.ic.ac.uk/software/neural/
3) Copy the ".../VeriNet/src" folder to ".../vnn-comp/verinet/"".
4) Install Gurobi as explained below. 
5) Run $chmod +x run_benchmarks to_tex
6) Run the bash script ".../vnn-comp/verinet/run_benchmarks".

The final results of all benchmarks are automatically compiled to a Latex pdf file .../vnn-comp/verinet/results.pdf

### Gurobi

VeriNet uses the Gurobi LP-solver which has a free academic license.  

1) Go to https://www.gurobi.com, download Gurobi and obtain the license.  
2) Follow the install instructions from http://abelsiqueira.github.io/blog/installing-gurobi-7-on-linux/  
3) Activate pipenv by cd'ing into your VeriNet/src and typing $pipenv shell
4) Find your python path by typing $which python
5) cd into your Gurobi installation and run $<your python path> setup.py install

## Authors

Patrick Henriksen: ph818@ic.ac.uk  
Alessio Lomuscio
