from src import utils
from src import parser
import os
import numpy as np
import math
import time


class SPOQ(object):
    def __init__(self,K=None,x=None,noise=None,sigma=None,y=None,J=5000, verbosity=0):
        self.input_path = os.path.join(os.getcwd(), "inputs")
        self.params = parser.ParamParser(os.path.join(self.input_path, "params.config"))
        self.K = K
        self.noise = noise
        self.x = x
        self.alpha = self.params.alpha
        self.beta = self.params.beta
        self.nbiter = self.params.nbiter
        self.metric = self.params.metric
        self.eta = self.params.eta
        self.p = self.params.p
        self.q = self.params.q
        self.sigma = sigma
        self.N = len(self.x)
        self.xi = 1.1 * math.sqrt(self.N) * self.sigma
        self.y = y
        self._xk = None
        self._fcost = None
        self._Bwhile = None
        self._Time = None
        self._mysnr = None
        self.J = J
        self.verbosity = verbosity

    def run(self):
        self._xk, self._fcost, self._Bwhile, self._Time, self._mysnr = utils.FB_PPXALpLq(self.K, self.y, self.p, self.q, self.metric,
                                                                                         self.alpha, self.beta, self.eta, self.xi,
                                                                                         self.nbiter, self.x, self.J, self.verbosity)

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

    @property
    def xtrue(self):
        return self.x
