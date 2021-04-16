"""This modules provides reusable classes and functions to read data and parameters as well
as to generate data for simulations
"""

import configparser
import numpy as np
import os
from scipy.linalg import pascal, toeplitz

def read_vector(path):
    """This function reads a vector from a file

    Parameters
    ----------
    path : str 
        the path to the file containing the vector values
    Returns
    -------
    array
        
    """
    with open(path, "r") as f:
        v = f.readlines()
        v = np.array([float(x[:-1]) for x in v])
    return v


def read_matrix(path):
    """This function reads a matrix from a file

    Parameters
    ----------
    path : str 
        the path to the file containing the matrix values
    Returns
    -------
    array
        
    """
    with open(path, "r") as f:
        K = f.readlines()
        K = np.array([x[:-1].split() for x in K], dtype=float)
    return K

class DataGenerator(object):
    """A class to generate data for simulation
    
    Attributes
    ----------
    nSample : int
        length of the original array to generate
    nPeak : int 
        number of values that should be different from 0 in the original array
    peakWidth : int

    """

    def __init__(self,nSample,nPeak,peakWidth):
        self._xtrue = np.zeros((nSample,1))
        xtrueLocation = np.random.choice(nSample, nPeak, replace=False)
        xtrueAmplitude = np.random.rand(nPeak).reshape(-1,1)
        self._xtrue[xtrueLocation,:] = xtrueAmplitude
        peakMatrix = pascal(peakWidth) 
        peakShape = np.diag(np.fliplr(peakMatrix))
        peakShape = peakShape/np.sum(peakShape)
        peakShape = peakShape.reshape(-1,1)
        peakShapeFilled = np.concatenate((peakShape,np.zeros((nSample-peakWidth,1))))
        self._K = toeplitz(peakShapeFilled)
        self._y = np.dot(self._K,self._xtrue)
        self._noise = np.random.normal(size=(nSample,1))
        self._sigma = 0.5*np.max(self._y)/100
        self._y = self._y + self._sigma*self._noise

    @property
    def xtrue(self):
        return self._xtrue
    @property
    def K(self):
        return self._K
    @property
    def y(self):
        return self._y
    @property
    def noise(self):
        return self._noise
    @property
    def sigma(self):
        return self._sigma

class DataReader(object):
    """A class to read data from the file "inputs" for running the algorithm on the paper data
    """
    def __init__(self):
        input_path = os.path.join(os.getcwd(), "inputs")
        self._xtrue = read_vector(os.path.join(input_path,"x"))
        self._K = read_matrix(os.path.join(input_path,"K"))
        self._noise = read_vector(os.path.join(input_path,"noise"))
        self._y =  np.dot(self._K,self._xtrue)
        self._sigma = 0.1*np.max(self._y)/100
        self._y = self._y + self._sigma*self._noise
        

    @property
    def xtrue(self):
        return self._xtrue
    @property
    def K(self):
        return self._K
    @property
    def y(self):
        return self._y
    @property
    def noise(self):
        return self._noise
    @property
    def sigma(self):
        return self._sigma


class ParamParser(object):
    """A class to parse the params.config file for setting the input values for the SPOQ class
    
    Attributes
    ----------
    path : str
        the path to the file containing params.config

    """
    def __init__(self, path):
        self.config = configparser.ConfigParser(strict=False)
        self.config.read(path)
        self._alpha = float(self.config["PARAMS"]["alpha"])
        self._beta = float(self.config["PARAMS"]["beta"])
        self._eta = float(self.config["PARAMS"]["eta"])
        self._metric = int(self.config["PARAMS"]["metric"])
        self._p = float(self.config["PARAMS"]["p"])
        self._q = float(self.config["PARAMS"]["q"])
        self._nbiter = int(self.config["PARAMS"]["nbiter"])
        

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
    def nbiter(self):
        return self._nbiter
