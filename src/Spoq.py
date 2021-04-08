import numpy as np
import utils
import math
import time
from numpy import linalg as LA


class FBPPXALpLq(object):

    def __init__(self,K,y,p,q,metric,alpha,beta,eta,xi,nbiter,xtrue):
        self.nbiter = nbiter
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.y = y
        self.p = p
        self.q = q
        self.K = K
        self.N = K.shape[-1]
        self.xi = xi
        self.xtrue = xtrue
        self.mysnr = np.zeros((nbiter,1))
        self.fcost = np.zeros((nbiter,1))
        self.Bwhile = np.zeros((nbiter,1))
        self.Time = []
        self.max_iter = 5000
        self.gamma = 1
        self.prec = 1e-12
        self.L = utils.ComputeLipschitz(alpha,beta,eta,p,q,self.N)
        self.metric = metric
        self.xk = utils.pds(K,y,xi,10)[0]
        self.mysnr[0] = -10*math.log10(np.sum((self.xk-xtrue)**2) / np.sum(xtrue**2))
        self.fcost[0] = utils.Fcost(self.xk,alpha,beta,eta,p,q)
    
    def run(self):
        for k in range(self.nbiter):
            xk_old = self.xk
            if k%100 == 0 :
                print("it = {!s} : fcost = {!s} \n".format(k,self.fcost[k-1]))
            
            start_time = time.time()
            if self.metric == 0 :
                A = self.L*np.ones((self.N,1))
                B = A / self.gamma
                xxk = self.xk - (1/B)*utils.gradlplq(self.xk,self.alpha,self.beta,self.eta,
                                            self.p,self.q)
                self.xk = utils.proxPPXAplus(self.K,B,xxk,self.y,self.xi,
                                            self.max_iter,self.prec)
            elif self.metric == 1:
                A = utils.condlplq(self.xk,self.alpha,self.beta,self.eta,
                                    self.p,self.q,0)           
                B = A / self.gamma
                xxk = self.xk - (1/B)*utils.gradlplq(self.xk,self.alpha,self.beta,
                                                self.eta,self.p,self.q)
                self.xk = utils.proxPPXAplus(self.K,B,xxk,self.y,self.xi,
                                    self.max_iter,self.prec)
            else:
                ro = np.sum(np.abs(self.xk**self.q))**(1/self.q)
                bwhile = 0
                while True: 
                    A = utils.condlplq(self.xk,self.alpha,self.beta,self.eta,
                                        self.p,self.q,ro)
                    B = A / self.gamma
                    xxk = self.xk - (1/B)*utils.gradlplq(self.xk,self.alpha,self.beta,
                                                self.eta,self.p,self.q)  

                    self.xk = utils.proxPPXAplus(self.K,B,xxk,self.y,self.xi,
                                    self.max_iter,self.prec)

                    if np.sum(np.abs(self.xk)**self.q)**(1/self.q) < ro :
                        ro = ro/2
                        bwhile = bwhile + 1
                    else :
                        break

                self.Bwhile[k]= bwhile
            end_time = time.time()
            self.Time.append(end_time - start_time)
            error = LA.norm(self.xk-xk_old)**2 / LA.norm(xk_old)**2
            self.mysnr[k+1] = -10*math.log10(np.sum((self.xk-self.xtrue)**2)/np.sum(self.xtrue**2))
            self.fcost[k+1] = utils.Fcost(self.xk,self.alpha,self.beta,self.eta,
                                            self.p,self.q)

            if error < self.prec:
                break
    
    def display_info(self):
        if self.Time:
            SNR = -10*math.log10(np.sum((self.xtrue-self.xk)**2)/np.sum(self.xtrue**2))
            print("Reconstruction in {0!s} iterations\n".format(len(self.Time)))
            print("SNR =  {0!s}\n".format(SNR))
            print("Reconstruction time is {0!s}s. \n".format(np.sum(self.Time)))



if __name__ == "__main__" :
    xtrue = utils.read_vector("../data/x")
    K = utils.read_matrix("../data/K")
    y = K*xtrue
    sigma = 0.1*np.max(y)/100
    noise = utils.read_vector('../data/noise')
    y = y + sigma*noise
    N = len(xtrue)
    xi = 1.1*math.sqrt(N)*sigma
    eta = 2E-6
    alpha = 7E-7
    beta = 3E-2
    p = 0.75
    q = 2
    nbiter=5000
    solver = FBPPXALpLq(K,y,p,q,2,alpha,beta,eta,xi,nbiter,xtrue)
    solver.run()
    solver.display_info()









        
            
            


                

