from __future__ import division
import numpy as np

train = open('IrisTrain2014.dt', 'r')
test = open('IrisTest2014.dt', 'r')

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

trainset = read_data(train)
testset = read_data(test)

"""
This function takes the dataset and separates the datapoints in lists according to class.
An array containing c numbers of arrays is returned - c is number of classes. 
"""
def separate_in_classes(dataset):
	temp = []
	separated = []
	cl =0.0
	while cl < 3.0:
		for datapoint in dataset:
			if datapoint[-1] == cl:
				temp.append(datapoint)
		cl += 1.0
		separated.append(temp)
		temp = []

	return np.array(separated)

"""
This function takes a (subset of a) dataset and computes the mean of each class (leaving the class column out)
it returns an array of s values - s is the number of features
"""
def mean(dataset):
	Mean = []
	number_of_features = len(dataset[0]) - 1 #Leaving out the class

	for i in xrange(number_of_features): 
		s = 0
		for elem in dataset:
			s += elem[i]
		mean = s / len(dataset)
		Mean.append(mean)
	return np.array(Mean)

"""
This function finds the between class scatter matrix using the class mean (1 mean per feature) and the overall mean (1 mean per feature)
It returns an N x N matrix - N is number of features
"""
def bwt_class_scatter_matrix(subset, dataset):
	classmean = mean(subset)
	overallmean = mean(dataset)
	number_of_features = len(dataset[0]) - 1 #Leaving out the class

	between = np.outer((classmean - overallmean),(classmean - overallmean))
	return np.matrix(between)

"""
This function find the within scatter matrix.
It takes a subset of the data. 
For every datapoint it sums the matrix up. It returns the summed matrix

"""
def wtn_class_scatter_matrix(subset):
	classmean = mean(subset)
	number_of_features = len(subset[0]) - 1 #Leaving out the class
	su = np.zeros(shape=(number_of_features,number_of_features))
	within = np.zeros(shape=(number_of_features,number_of_features))

	for datapoint in subset:
		su += np.outer((datapoint[:number_of_features] - classmean), (datapoint[:number_of_features] - classmean))
	within +=su
	return np.matrix(within)

train_separated = separate_in_classes(trainset)
wtn_class_scatter_matrix(train_separated[0])
bwt_class_scatter_matrix(train_separated[0], trainset)