./evaluation/mnist.sh
./evaluation/acas.sh
./evaluation/acas-hard.sh
python3 ./evaluation/tools/tex.py
pdflatex evaluation/latex/results.tex
