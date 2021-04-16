from src import spoq
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from src import parser
import argparse



if __name__ == "__main__" :

    argsparser = argparse.ArgumentParser()

    data = parser.DataGenerator(500,20,5)
    argsparser.add_argument("-s", "--simulation",action='store_true', help="If True the algorithm will run on simulated data")
    argsparser.add_argument("-nsample", "--n",default=500,type =int, help="The length of x")
    argsparser.add_argument("-npeak", "--np",default=20,type =int, help="The length of the values different than 0 in x")
    argsparser.add_argument("-peakw", "--pw",default=5,type =int, help="The peak width")
    args = argsparser.parse_args()

    if args.simulation :
        data = parser.DataGenerator(args.n,args.np,args.pw)
    else :
        data = parser.DataReader()


    solver = spoq.SPOQ(x=data.xtrue,y=data.y,K=data.K,sigma=data.sigma,noise=data.noise)
    print("Running TR-VMFB algorithm on SPOQ penalty with p = {0!s} and q = {1!s}\n".format(solver.p,solver.q))
    start_time = time.time()
    solver.run()
    end_time = time.time()
    print("Reconstruction in {0!s} iterations\n".format(len(solver.Time)))
    print("SNR = {0!s}\n".format(-10*math.log10(np.sum((solver.xtrue-solver.xk)**2)/np.sum(solver.xtrue**2))))
    print("Reconstruction time is {0!s} s.".format(np.sum(solver.Time)))

    # Plot results
    x_time = np.cumsum(solver.Time)
    x_time = np.insert(x_time,0,0)
    fig, ax = plt.subplots()
    ax.plot(solver.xtrue,"ro",label = "Original signal")
    ax.plot(solver.xk,"bo", label = "Estimated signal")
    ax.legend()
    ax.set_title("Reconstruction results")

    fig2, ax2 = plt.subplots()
    ax2.plot(x_time,solver.mysnr,'-k',label="TR-VMFB")
    ax2.set_title("Algorithm convergence")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("SNR (dB)")
    ax2.legend()
    plt.show()
    



    

    






    