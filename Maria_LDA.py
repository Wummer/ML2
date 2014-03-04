from __future__ import division
import numpy as np
import math
import operator

train = open('IrisTrain2014.dt', 'r')
test = open('IrisTest2014.dt', 'r')

##############################################################################
#
#                             Preparing  
#
##############################################################################
"""
This function reads in the files, strips by newline and splits by space char. 
It returns the dataset as numpy arrays.
"""
def read_data(filename):
	data_set = ([])
	for l in filename.readlines():
		l = np.array(l.rstrip('\n').split(),dtype='float')
		data_set.append(l)
	return data_set


"""
This function expects a dataset and splits the labels and the feature into two separate lists.
The feature vector is reshaped to a column vector. 
The labels in the label list has the same index as the adjacent feature vector in the feature list.
Both lists are returned. 
"""
def prepare_dataset(dataset):
	labels=[]
	features = []
	for datapoint in dataset:
		labels.append(datapoint[-1])
		features.append(datapoint[:-1].reshape(2,1))
	return features, labels


"""
This function expects the dataset and separates the datapoints in lists according to class.
It takes out the class label (that is now reflected by list structure) and saves that in a list of lists of the same structure as the feature values.
It returns and array of one array per class and a list of lists for the classlabels. 
"""
def separate_in_classes(dataset):
	temp = []
	separated = []
	classtemp = []
	classsep = []

	cl =0.0
	while cl < 3.0:
		for datapoint in dataset:
			if datapoint[-1] == cl:
				datapoint = datapoint[:2].reshape(2,1)
				temp.append(datapoint[:2])
				classtemp.append(datapoint[-1])
		cl += 1.0
		separated.append(temp)
		classsep.append(classtemp)
		temp = []

	return np.array(separated), classsep




##############################################################################
#
#                             Normalizing 
#
##############################################################################
"""
This function takes a dataset with class labels and computes the mean and the variance of each input feature (leaving the class column out)
It returns two lists: [mean of first feature, mean of second feature] [variance of first feature, variance of second feature]
"""
def mean_variance(data):
	Mean = []
	Variance = []
	number_of_features = len(data[0]) - 1 #Leaving out the class
	for i in xrange(number_of_features): 
		s = 0
		su = 0

		#mean
		for elem in data:
			s +=elem[i]
		mean = s / len(data)
		Mean.append(mean)
		
		#variance:
		for elem in data:
			su += (elem[i] - Mean[i])**2
			variance = su/len(data)	
		Variance.append(variance)
	return Mean, Variance


"""
This function expects a dataset with class labels.
It calls mean_variance to get the mean and the variance for each feature
Then these values are used to normalize every datapoint to zero mean and unit variance.
A copy of the data is created. 
The normalized values are inserted at the old index in the copy thus preserving class label 
The new, standardized data set with untouched class labels is returned
"""
def meanfree(data):
	number_of_features = len(data[0]) - 1 #Leaving out the class
	mean, variance = mean_variance(data)

	new = np.copy(data)
	for i in xrange(len(new)):
		for num in xrange(number_of_features):
			#replacing at correct index in the copy
			new[i][num] = (new[i][num] - mean[num]) / np.sqrt(variance[num])
	return new 

"""
This function transforms the test set using the mean and variance from the train set.
It expects the train and test set including class labels. 
It makes a copy of the test sets and inserts the transformed feature value at the correct index.
It returns the transformed test set woth untouched labels. 

"""
def transformtest(trainset, testset):
	#getting the mean and variance from train:
	meantrain, variancetrain = mean_variance(trainset)
	number_of_features = len(trainset[0]) - 1 #Leaving out the class
	newtest = np.copy(testset)

	for num in xrange(number_of_features):
		for i in xrange(len(testset)):
			#replacing at correct index in the copy
			newtest[i][num] = (testset[i][num] - meantrain[num]) / np.sqrt(variancetrain[num])
	return newtest


##############################################################################
#
#                           Variables for LDA
#
##############################################################################

"""
This function expects a subset (an entire class). 
It calculates the mean of each feature in each class 
and returns an array with one array per class with this class' mean. 
"""
def class_mean(separated):
	common = []
	classmean=[]
	for subset in separated:
		mean = sum(subset) / len(subset)
		newmean = mean.reshape(1,2)
		classmean.append(newmean[0])
	common.append(classmean)
	new = np.asarray(common[0])
	return new


"""

This function expects the dataset separated in lists according to class.
It calculates the covariance and sums it up and divides by number of datapoints - number of classes.
It returns the summed matrix, which is N x N matrix - N is number of features

"""
def cova(separated):
	N = sum(len(x) for x in train_separated)
	number_of_features = len(separated[0][0]) 
	common = np.zeros(shape=(number_of_features, number_of_features))
	su = np.zeros(shape=(number_of_features,number_of_features))
	within = np.zeros(shape=(number_of_features,number_of_features))

	for subset in separated:
		classmean = sum(subset) / len(subset)	
		for datapoint in subset:
			su += np.outer((datapoint - classmean), (datapoint - classmean).T)
		within +=su
	common += within
	common = np.matrix(common)
	common = common / (N - len(separated))
	return common

def prior(separated, dataset):
	#getting the prior
	Y = []
	for subset in train_separated:
		y = len(subset) / len(dataset)
		Y.append(y)
	return np.asarray(Y)

##############################################################################
#
#                                 LDA
#
##############################################################################

"""
In this function the decision boundary is found. 
It expects a dataset (2D array), adjacent 1D list of labels and a train set that is separated in lists according to classmean
It calls functions to calculate the covariance and the classmean on the separated set.
It makes one array Y of prior probabilities for each class. 
Then it finds the argmax and compares the prediced label to the actual label and counts up to get the error. 
"""
def LinDA(dataset, labels, train_separated):
	cov = cova(train_separated)
	classmean = class_mean(train_separated)

	Y = prior(train_separated, dataset)
	
	counter = 0
	for i in xrange(len(dataset)): 		
		deltas = []
		for k in xrange(len(train_separated)): #for every class
			deltaofx = np.dot(np.dot(dataset[i].T, cov.I), classmean[k]) -  0.5*(np.dot(np.dot(classmean[k].T, cov.I), classmean[k])) + math.log(Y[k])
			deltas.append(deltaofx)
		#finding argmax
		max_index, max_value = max(enumerate(deltas),key=operator.itemgetter(1)) #finding index of min value
		max_index = float(max_index)
		if max_index != labels[i]:
			counter += 1
		
	error = counter / len(dataset)
	print "Error" , error


##############################################################################
#
#                             Printing 
#
##############################################################################

#Common
trainset = read_data(train)
testset = read_data(test)

print "*" * 45
print "LDA"
print "*" * 45

#For non-standardized
print "-" * 45
print "Not standardized"
print "-" * 45
train_separated, trainclasses = separate_in_classes(trainset)

testfeatures, testlabels = prepare_dataset(testset)
trainfeatures, trainlabels = prepare_dataset(trainset)

print "Trainset:"
LinDA(trainfeatures, trainlabels, train_separated)
print "Testset:"
LinDA(testfeatures, testlabels, train_separated)


# For standardized
print "-" * 45
print "Standardized"
print "-" * 45
stan_train = meanfree(trainset)
transf_test = transformtest(trainset, testset)

stan_train_separated, trainclasses = separate_in_classes(stan_train)

stan_train_feat, trainlabels = prepare_dataset(stan_train)
stan_test_feat, testlabels = prepare_dataset(transf_test)

print "Trainset:"
LinDA(stan_train_feat, trainlabels, stan_train_separated)
print "Testset:"
LinDA(stan_test_feat, testlabels, stan_train_separated)

