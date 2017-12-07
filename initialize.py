import numpy as np

def init(X, num_layers, hidden, num_epochs):

	#np.random.seed(256)
	num_trainset = X.shape[0]

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

	loss = np.zeros((num_epochs, 1))

	return out, net_in, net_in_bias, theta, error, dtheta, loss