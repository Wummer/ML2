from __future__ import division
import numpy as np
import pylab as plt
import math

train = open('IrisTrain2014.dt', 'r')
test = open('IrisTest2014.dt', 'r')

"""
This function reads ind in the files, strips by newline and splits by space char. 
It returns the dataset as numpy arrays.
"""
def read_data(filename):
	data_set = ([])
	for l in filename.readlines():
		l = np.array(l.rstrip('\n').split(),dtype='float')
		data_set.append(l)	
	return data_set

trainset = read_data(train)
testset = read_data(test)


#Computing mean and variance
"""
This function takes a dataset and computes the mean and the variance of each input feature (leaving the class column out)
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
This function calls mean_variance to get the mean and the variance for each feature
Then these values are used to normalize every datapoint to zero mean and unit variance.
A copy of the data is created. 
The normalized values are inserted at the old index in the copy thus preserving class label 
The new, standardized data set is returned
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
This function transforms the test set using the mean and variance from the train set
Assuming that both test and train set are i.i.d. from the same distribution this will make the
test set mean free unit variance.  
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



"""
This function expects the dataset and separates the datapoints in lists according to class.
It takes out the class label (that is now reflected by list structure) and saves that in a list of lists of the same structure as the feature values.
It returns and array of c numbers of arrays - c is number of classes and the same structure for the classes. 
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

"""
This function expects a dataset or a subset and computes the mean of the dataset (leaving the class column out)
it returns an array of s values - s is the number of features
"""
def mean(dataset):
	mean = ([0,0])

	mean = sum(dataset) / len(dataset)
	return mean


train_separated, trainclasses = separate_in_classes(trainset)
N = sum(len(x) for x in train_separated)

"""
This function expects a subset (an entire class) and the entire dataset. 
This function finds the between class scatter matrix using the class mean (1 mean per feature) and the overall mean (1 mean per feature)
It returns an N x N matrix - N is number of features
"""

def class_mean(separated):
	number_of_features = len(separated[0][0]) #Leaving out the class
	common = []

	for subset in separated:
		
		classmean = mean(subset)
		common.append(classmean)
	return common

"""
This function calculates the within class scatter matrix for all classes.
It expects a subset of the data. 
For every datapoint it sums the feature values up. 
It returns the summed matrix, which is N x N matrix - N is number of features

"""
def cova(separated):
	number_of_features = len(separated[0][0]) #Leaving out the class
	common = np.zeros(shape=(number_of_features, number_of_features))
	su = np.zeros(shape=(number_of_features,number_of_features))
	within = np.zeros(shape=(number_of_features,number_of_features))

	for subset in separated:
		classmean = mean(subset)	
		for datapoint in subset:
			su += np.outer((datapoint - classmean), (datapoint - classmean))
		within +=su
	common += within
	common = np.matrix(common)

	common = common / (N - len(separated))
	return common

def LDA(separated):
	cov = cova(separated)
	classmean = class_mean(separated)
	
	for k in separated:
		for datapoint in k: 
			deltaofx = np.dot(np.dot(datapoint.T, cov.I), classmean) -  (np.dot(np.dot(classmean.T, cov.I), classmean))/2 + math.log(len(k) / N)
			print deltaofx
			



LDA(train_separated)

# normalizing
standardized_train = meanfree(trainset)
transformed_test = transformtest(trainset, testset)
m,v = mean_variance(transformed_test)

# mu k is a vector of all 3 k's

#eigw, eigv = np.linalg.eig(cov1)
"""
plt.plot([elem[0] for elem in train_separated[0]],[elem[1] for elem in train_separated[0]],'x')
plt.plot([elem[0] for elem in train_separated[1]],[elem[1] for elem in train_separated[1]],'ro')
plt.show()
"""