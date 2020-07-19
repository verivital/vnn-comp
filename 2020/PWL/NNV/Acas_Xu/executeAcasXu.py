import matlab.engine
import os
import argparse

def executeAcasXu(filePath, property_id,n1,N1,n2,N2, method,numCores, t_out, hard):
    # start matlab engine
    eng = matlab.engine.start_matlab()

    eng.addpath(filePath, nargout=0)

    #N1 = 5;
    #N2 = 9;

    safes_num = eng.ones(N1, N2, nargout = 1);
    vts = eng.zeros(N1, N2, nargout = 1);
    total_vt = 0;
    n_safe = 0;
    n_unsafe = 0;
    numCores = 4
    print("\n====================Property %d=====================" % (property_id));
    for i in range (n1,N1+1):
        for j in range (n2,N2+1):
            print("\n====================Network %d_%d=====================" % (i, j));
            future = eng.verifyNetwork(i, j, property_id, method, numCores,nargout=2,background=True);
            try:
                if i==1 and j==1:
                    t_out1 = t_out+19 #for the first run it takes around 19 secs to run the matlab engine
                else:
                    t_out1 = t_out
                safe,vt = future.result(timeout=t_out1)
            except matlab.engine.TimeoutError:
                print("timeout")
                future.cancel()
                safe = 3
                vt = t_out
            safes_num[i-1][j-1]=int(safe)
            vts[i-1][j-1]=vt
    eng.printResultsACASXu(n1,n2,N1,N2,property_id,method,numCores,safes_num,vts,total_vt,n_safe,n_unsafe,hard,nargout=0)

def main():        
    parser = argparse.ArgumentParser(description ='script to execute ACASXu properties')
    parser.add_argument('filePath')
    parser.add_argument('property_id',type = int)
    
    parser.add_argument('n1',type = int, default = 1)
    parser.add_argument('N1',type = int, default = 5)
    parser.add_argument('n2',type = int, default = 1)
    parser.add_argument('N2',type = int, default = 9)

    
    parser.add_argument('numCores', type = int, default = 4)
    parser.add_argument('t_out', type = int, default = 300)
    parser.add_argument('hard', type = int, default = 0)

    args = parser.parse_args()
    executeAcasXu(args.filePath, args.property_id, args.n1, args.N1, args.n2, args.N2,'exact-star', args.numCores, args.t_out, args.hard)

if __name__ == "__main__":
    main()
