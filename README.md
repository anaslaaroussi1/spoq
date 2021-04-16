# SPOQ


## Project Description

The goal of the project is to reimplement the SPOQ-Sparse-Restoration-Toolbox-v1.0 from Matlab to python

## Files Description 

.
 * [inputs](./inputs)
   * [x](./inputs/x) : Original data
   * [K](./inputs/K) : Observation operator
   * [noise](./inputs/noise) : noise data
   * [y](./inputs/y) : measurment data
   * [params] (./inputs/params.config) : Hyperparameters of the algorithm
 * [src](./src) : The source code of the SPOQ library
 * [docs](./docs) : The documentation (static files for website deployment)
 

## Dependencies 

* [numpy](https://numpy.org/)
* [Matplotlib](https://matplotlib.org/)
* [scipy](https://www.scipy.org/)


## Usage 

### Install Dependencies 

> pip install numpy

> pip install matplotlib

> pip install scipy



### Running The algorithm 

The algorithm can run in two modes either from files in the input directory (x,K,y,noise) or by generating a random sparse signal for simulation purposes. The simulated signal is caracterized by :
* nsamples : number of samples 
* npeaks : number of samples different than 0
* peakw : the peaks width

First you have to install the dependencies and clone the project : 

> git clone https://gitlab-student.centralesupelec.fr/Anas.Laaroussi/spoq.git

To run the code from on inputs from inputs folder, in the root directory of the project run:

> python main.py 

To run the code on simulated signal with default values (nsample = 500, npeak = 20, peakw = 5) 

> python main.py -s

To change the caracterisitcs of the signal, let's say nsamples = 1000, npeak = 50 and peakw = 7 :

> python main.py -s -nsample 1000 -npeak 50 -peakw 7

## Documentation

The documentation of the project is to be find here [doc](https://anas.laaroussi.pages-student.centralesupelec.fr/spoq/utils.html)


  

   



