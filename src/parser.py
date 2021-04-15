import configparser
import numpy as np
import os

def read_vector(path):
    with open(path,"r") as f:
        v = f.readlines()
        v = np.array([float(x[:-1]) for x in v])
    return v


def read_matrix(path):
    with open(path,"r") as f:
        K = f.readlines()
        K = np.array([x[:-1].split() for x in K],dtype=float)
    return K

class ParamParser(object):
    def __init__(self,path):
        self.config = configparser.ConfigParser(strict=False)
        self.config.read(path)
        self._alpha = float(self.config["PARAMS"]["alpha"])
        self._beta = float(self.config["PARAMS"]["beta"])
        self._eta = float(self.config["PARAMS"]["eta"])
        self._metric = int(self.config["PARAMS"]["metric"])
        self._p = float(self.config["PARAMS"]["p"])
        self._q = float(self.config["PARAMS"]["q"])
        self._nbiter = int(self.config["PARAMS"]["nbiter"])
        self._K = self.config["PARAMS"]["K"]
        self._x = self.config["PARAMS"]["x"]
        self._y = self.config["PARAMS"]["y"]
        self._noise = self.config["PARAMS"]["noise"]

    @property
    def alpha(self):
        return self._alpha
    @property
    def beta(self):
        return self._beta
    @property
    def eta(self):
        return self._eta
    @property
    def metric(self):
        return self._metric
    @property
    def p(self):
        return self._p
    @property
    def q(self):
        return self._q
    @property
    def K(self):
        return self._K
    @property
    def x(self):
        return self._x
    @property
    def y(self):
        return self._y
    @property
    def noise(self):
        return self._noise
    @property
    def nbiter(self):
        return self._nbiter
    


