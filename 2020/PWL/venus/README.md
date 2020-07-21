# Running the benchmarks

1) Download Venus from https://vas.doc.ic.ac.uk/software/neural/
2) Copy .../venus-1.0/src, .../venus-1.0/_____main__.py and .../venus-1.0/resources to .../vnn-comp/2020/PWL/venus
3) Install python requirements: pip3 install -r .../vnn-comp/2020/PWL/venus/requirements.txt 
4) Install Gurobi:
	4a) wget https://packages.gurobi.com/9.0/gurobi9.0.1_linux64.tar.gz
	4b) tar xvfz gurobi9.0.1_linux64.tar.gz
	4c) cd gurobi901/linux64
	4d) python3 setup.py install
5) Run the benchmakrs: 
	5a) cd .../vnn-comp/2020/PWL/venus
	5b) ./run_benchmarks.sh

The results will be reported in the file "results.pdf" in .../vnn-comp/2020/PWL/venus

