import numpy as np
from evaluation.tools.tex_util import plot
import os 

directory = 'evaluation/latex'
if not os.path.exists(directory):
    os.makedirs(directory)


def parse_acas(logfile):
    statuses = []
    times = []
    f = open(logfile,'r')
    f.readline()
    for line in f:
        tokens = line.split(',')
        if tokens[2] == 'Satisfied':
            statuses.append('Safe')
        elif tokens[2] == 'NOT Satis':
            statuses.append('Unsafe')
        else:
            statuses.append('Undecided')
        times.append(float(tokens[3]))

    return np.array(statuses,dtype='str'), np.array(times,dtype='float')

def parse_mnist(logfile):
    statuses = []
    times = []
    f = open(logfile,'r')
    f.readline()
    for line in f:
        tokens = line.split(',')
        if tokens[3] == 'Satisfied':
            statuses.append('Safe')
        elif tokens[3] == 'NOT Satis':
            statuses.append('Unsafe')
        else:
            statuses.append('Undecided')
        times.append(float(tokens[4]))
    return np.array(statuses,dtype='str'), np.array(times,dtype='float')




def write_acas_table(logfile, outfile, caption):
    with open(outfile,'w') as f:
        f.write('\\begin{longtable}{llll}\n')
        f.write(f'  \\caption{{{caption}}}\\\\\n')
        f.write('   \\toprule\n')
        f.write('    Property & Network &  Result & Time \\\\\n')

        l = open(logfile,'r')
        l.readline()
        prop_six = 0
        for line in l:
            tokens = line.split(',')
            net = tokens[0][5] + '-' + tokens[0][7]
            prop = tokens[1].split(' ')[1]
            if prop=='6a':
                prop_six = float(tokens[3])
                continue
            if tokens[2] == 'Satisfied':
                result = 'Satisfied'
            elif tokens[2] == 'NOT Satis':
                result = 'Unsatisfied'
            else:
                result = 'Undecided'
            time =  float(tokens[3])
            if prop=='6b':
                prop = '6'
                time += prop_six
            if time > 300:
                time = '-'

            f.write(f'    {prop} & {net} & {result} & {time} \\\\\n')
        f.write('    \\bottomrule\n')
        f.write('\\end{longtable}\n\n')



def write_mnist_table(logfile, outfile):
    with open(outfile,'w') as f:
        f.write('\\begin{longtable}{lllll}\n')
        f.write('  \\caption{Total runtime (sec) for MNIST-ALL}\\\\\n')
        f.write('   \\toprule\n')
        f.write('    Network & Radius & Image &  Result & Time   \\\\\n')
        f.write('    \\midrule \n')

        l = open(logfile,'r')
        l.readline()
        for line in l:
            tokens = line.split(',')
            if tokens[0] == 'mnist-net_256x2.onnx':
                net = 'MNIST-256x2'
            elif tokens[0] == 'mnist-net_256x4.onnx':
                net = 'MNIST-256x4'
            else:
                net = 'MNIST-256x6'
            im = tokens[1][5:]
            radius = tokens[2]
            if tokens[3] == 'Satisfied':
                result = 'Satisfied'
            elif tokens[3] == 'NOT Satis':
                result = 'Unsatisfied'
            else:
                result = 'Undecided'
            time =  float(tokens[4])
            f.write(f'    {net} & {radius} & {im} & {result} & {time} \\\\\n')
        f.write('    \\bottomrule\n')
        f.write('\\end{longtable}\n\n')




statuses, times = parse_acas('evaluation/acas.log')
plot(statuses, times, 'ACASXU-ALL', 'evaluation/latex/acasxu-all.pdf', 300)
write_acas_table('evaluation/acas.log','evaluation/latex/acasxu-all.tex', 'Total runtime (sec) for ACASXU-ALL')
write_acas_table('evaluation/acas-hard.log','evaluation/latex/acasxu-hard.tex', 'Total runtime (sec) for ACASXU-HARD')

statuses, times = parse_mnist('evaluation/mnist.log')
plot(statuses, times, 'PAT-FCN', 'evaluation/latex/mnist.pdf', 900)
write_mnist_table('evaluation/mnist.log','evaluation/latex/mnist.tex')

