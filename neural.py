import numpy as np
import argparse

# class Batch:
	
# 	def __init__(self, X, Y):
		
# 		self.X = X
# 		self.Y = Y

# class Weight:

# 	def __init__(self, num_row, num_col):

# 		self.num_row = num_row
# 		self.num_col = num_col

# class Network:

# 	def __init__(self):

def sigmoid(x):
	return 1. / (1 + np.exp(-x))


X = np.random.randn(10, 3)
Y = np.random.randn(10, 3)

alpha = 0.1

num_layers = 3
num_trainset = 10

hidden = [3, 4, 3]
out = []
net_in = []
net_in_bias = []
theta = []
error = []
dtheta = []

#Net Input, Activation
for i in xrange(num_layers): # 0 1 2
	temp = np.zeros((num_trainset, hidden[i]))
	out.append(temp)
	net_in.append(temp)

out[0] = X # Input

#Error
for i in xrange(num_layers): # 0 1 2
	temp = np.zeros((hidden[i]+1, 1)) #error[0] not to be used
	error.append(temp)

#Bias Appended net_in
for i in xrange(num_layers): # 0 1 2
	temp = np.zeros((num_trainset, hidden[i]+1))
	net_in_bias.append(temp)

#Theta
theta.append(0) # Dummy
for i in xrange(num_layers-1): # 0 1
	temp = np.random.randn(hidden[i]+1, hidden[i+1])
	theta.append(temp) #Theta indices: 1 2 ...

#dtheta
dtheta.append(0) # Dummy
for i in xrange(num_layers-1): # 0 1
	temp = np.random.randn(hidden[i]+1, hidden[i+1])
	dtheta.append(temp) #dtheta indices: 1 2 ...

#Forward Propagation
for i in range(1, num_layers): # 1 2

	net_in_bias[i-1] = np.append(out[i-1], np.ones((num_trainset, 1)), axis = 1) 
	net_in[i] = np.matmul(net_in_bias[i-1], theta[i])
	out[i] = sigmoid(net_in[i])

# print out[num_layers-1].shape

#Backpropagation

#Error Calculation
for i in xrange(num_layers-1, 0, -1): # 2 1 

	if i == num_layers-1:
		error[i] = np.mean(out[i] * (1 - out[i]) * (Y - out[i]), axis = 0).reshape(hidden[i], 1)
	else:
		error[i] = np.mean(net_in_bias[i] * (1 - net_in_bias[i]) * np.transpose(np.matmul(theta[i+1], error[i+1])), axis = 0).reshape(hidden[i]+1, 1)
		# Ignore last node

# print error[1].shape

#Weight Update
for i in xrange(num_layers-1, 0, -1):

	if i == num_layers-1:
		dtheta[i] = np.matmul(np.transpose(net_in_bias[i-1]), np.repeat(error[i].reshape(1, hidden[i]), num_trainset, axis = 0)) * alpha
	else:
		dtheta[i] = np.matmul(np.transpose(net_in_bias[i-1]), np.repeat(error[i][:-1].reshape(1, hidden[i]), num_trainset, axis = 0)) * alpha

# num of inputs: int
# num of outputs: int 
# num of hidden layers: int 
# num of nodes in each hidden layer: integer list
# num of training samples