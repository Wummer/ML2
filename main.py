from __future__ import division
import numpy
from matplotlib import pylab as plt


"""
Part 1
"""
train = open('IrisTrain2014.dt', 'r')
test = open('IrisTest2014.dt', 'r')

#Calling read and split
train_set = LDA.read_data(train)
test_set = LDA.read_data(test)

