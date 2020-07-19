import matlab.engine
import os
import argparse

def executeImperial(filePath, netfile, epsilon, reachMethod, n, t_out, rF, numLayer):
    # start matlab engine
    eng = matlab.engine.start_matlab()
    eng.addpath(filePath, nargout=0)
    
    rb = eng.zeros(1, n, nargout = 1);
    vt = eng.zeros(1, n, nargout = 1);

    netfile = filePath + '/' + netfile

    for i in range(1,n+1):
        print("\n====================Image %d =====================" % (i));
        future = eng.verifyMnist_Imperial(netfile, epsilon,i, reachMethod, rF,numLayer, nargout=2,background=True);

        try:
            rb1,vt1 = future.result(timeout=t_out)
            rb[0][i-1] = rb1
            vt[0][i-1] = vt1
        except matlab.engine.TimeoutError:
            print("timeout")
            future.cancel()
            eng.quit()
            eng = matlab.engine.start_matlab()
            eng.addpath(filePath, nargout=0)
            rb[0][i-1] = 3
            vt[0][i-1] = t_out
    eng.saveDataMnist_Imperial(epsilon,rb,vt,numLayer,nargout=0)
          
def main():
    parser = argparse.ArgumentParser(description ='script to execute CIFAR 10')
    parser.add_argument('filePath')
    parser.add_argument('netfile')
    parser.add_argument('epsilon', type = float)
    parser.add_argument('t_out', type = int, default = 900)
    parser.add_argument('relaxFactor', type = float, default = 1)
    parser.add_argument('numLayer', type = int, default = 2)
    args = parser.parse_args()
    executeImperial(args.filePath, args.netfile, args.epsilon, 'approx-star',25, args.t_out,args.relaxFactor, args.numLayer)

if __name__ == "__main__":
    main()