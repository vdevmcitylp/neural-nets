import numpy as np

def sigmoid(x):
	return 1. / (1 + np.exp(-x))

def forward(num_layers, num_trainset, net_in_bias, out, net_in, theta):
	
	#Forward Propagation
	for i in range(1, num_layers): # 1 2

		net_in_bias[i-1] = np.append(out[i-1], np.ones((num_trainset, 1)), axis = 1) 
		net_in[i] = np.matmul(net_in_bias[i-1], theta[i])
		out[i] = sigmoid(net_in[i])

	return out, net_in, net_in_bias