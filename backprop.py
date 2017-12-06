import numpy as np

def back(num_layers, num_trainset, error, out, hidden, net_in_bias, theta, dtheta, alpha, Y):

	for i in xrange(num_layers-1, 0, -1): # 3 2 1 

		if i == num_layers-1:
			error[i] = np.mean(out[i] * (1 - out[i]) * (Y - out[i]), axis = 0).reshape(hidden[i], 1)
		elif i == num_layers-2:
			error[i] = np.mean(net_in_bias[i] * (1 - net_in_bias[i]) * np.transpose(np.matmul(theta[i+1], error[i+1])), axis = 0).reshape(hidden[i]+1, 1)
			# Ignore last node
		else:
			error[i] = np.mean(net_in_bias[i] * (1 - net_in_bias[i]) * np.transpose(np.matmul(theta[i+1], error[i+1][:-1])), axis = 0).reshape(hidden[i]+1, 1)

	#Gradient Calculation
	for i in xrange(num_layers-1, 0, -1):

		if i == num_layers-1:
			dtheta[i] = np.matmul(np.transpose(net_in_bias[i-1]), np.repeat(error[i].reshape(1, hidden[i]), num_trainset, axis = 0)) * alpha
		else:
			dtheta[i] = np.matmul(np.transpose(net_in_bias[i-1]), np.repeat(error[i][:-1].reshape(1, hidden[i]), num_trainset, axis = 0)) * alpha

	#Weight Update
	for i in xrange(1, num_layers):
		theta[i] += dtheta[i]

	return error, dtheta, theta