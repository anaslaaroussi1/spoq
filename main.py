from src import spoq
import time
import math
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__" :

    # Display Running information
    solver = spoq.SPOQ()
    print("Running TR-VMFB algorithm on SPOQ penalty with p = {0!s} and q = {1!s}\n".format(solver.p,solver.q))
    start_time = time.time()
    solver.run()
    end_time = time.time()
    print("Reconstruction in {0!s} iterations\n".format(len(solver.Time)))
    print("SNR = {0!s}\n".format(-10*math.log10(np.sum((solver.xtrue-solver.xk)**2)/np.sum(solver.xtrue**2))))
    print("Reconstruction time is {0!s} s.".format(np.sum(solver.Time)))

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(solver.xtrue,"ro")
    ax.plot(solver.xk,"bo")
    ax.set_title("Reconstruction results")

    






    