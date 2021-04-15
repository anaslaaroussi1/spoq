import utils
import parser
import os
import numpy as np
import math
import time


class SPOQ(object):
    def __init__(self):
        self.input_path = os.path.join(os.path.dirname(os.getcwd()),"inputs")
        self.params = parser.ParamParser(os.path.join(self.input_path,"params.config"))
        self.K = parser.read_matrix(os.path.join(self.input_path,self.params.K))
        self.noise = parser.read_vector(os.path.join(self.input_path,self.params.noise))
        self.x = parser.read_vector(os.path.join(self.input_path,self.params.x))
        self.y = parser.read_vector(os.path.join(self.input_path,self.params.y))
        self.alpha = self.params.alpha
        self.beta = self.params.beta
        self.nbiter = self.params.nbiter
        self.metric = self.params.metric
        self.eta = self.params.eta
        self.p = self.params.p
        self.q = self.params.q
        self.sigma = 0.1*np.max(self.y)/100
        self.N = len(self.x)
        self.xi = 1.1*math.sqrt(self.N)*self.sigma
        self._xk = None
        self._fcost = None
        self._Bwhile = None
        self._Time = None
        self._mysnr = None
    
    def run(self):
        self._xk,self._fcost,self._Bwhile,self._Time,self._mysnr = utils.FB_PPXALpLq(self.K,
                        self.y,self.p,self.q,self.metric,self.alpha,self.beta,self.eta,
                        self.xi,self.nbiter,self.x)
    @property 
    def xk(self):
        return self._xk
    @property
    def fcost(self):
        return self._fcost
    @property
    def Bwhile(self):
        return self._Bwhile
    @property
    def Time(self):
        return self._Time
    @property
    def mysnr(self):
        return self._mysnr


if __name__ == "__main__" :

    s = SPOQ()
    s.run()
    print(s.xk)

        

    






