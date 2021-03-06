import numpy as np


#Assumes classes start from 0

def label_binarize(Y, classes):

	num_trainset = Y.size
	num_class = len(classes)

	Y_one_hot = np.zeros((num_trainset, num_class))

	for i in xrange(num_trainset):
		Y_one_hot[i, int(Y[i])] = 1

	return Y_one_hot

# For classes = [1, 2, 3, ...]
# def label_binarize(Y, classes):

# 	num_trainset = Y.size
# 	num_class = len(classes)

# 	Y_one_hot = np.zeros((num_trainset, num_class+1))

# 	for i in xrange(num_trainset):
# 		Y_one_hot[i, int(Y[i])] = 1

# 	return Y_one_hot[:, 1:num_class+1]