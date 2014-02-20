from __future__ import division
import numpy
from matplotlib import pylab as plt

"""
This function reads ind in the files, strips by newline and splits by space char. 
It returns he dataset as numpy arrays.
"""
def read_data(filename):
	data_set = ([])
	for l in filename.readlines():
		l = np.array(l.rstrip('\n').split(),dtype='float')
		data_set.append(l)	
	return data_set

def Mean(x,y):
	MLx = sum(x)*1/len(x)
	MLy = sum(y)*1/len(y)

	return MLx,MLy

def Sigma_ML(x,y,ML):
	assert len(x) == len(y)
	samples = []
	nM  = 0

	for i in range(len(x)):
		samples.append(np.array([x[i],y[i]]).reshape(2,1)) #2 columns, 1 row, i.e. vector plots
	samples = np.array(samples)

	for i in range(len(x)):
		n = samples[i]-ML
		nM += np.dot(n,n.T)
	CML = (1/len(x))*nM

	return CML
