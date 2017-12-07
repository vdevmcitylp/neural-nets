import numpy as np
import read_data, label_binarize, performance_metrics, initialize, forwardprop, backprop, plot
import matplotlib.pyplot as plt

#np.random.seed(42)

X, Y_nb = read_data.input()

X = X.reshape(X.shape[0], X.shape[1])
Y = label_binarize.label_binarize(Y_nb, classes = [0, 1])

num_trainset = X.shape[0]

alpha = 0.0001

# UPDATE THESE
num_class = 2
num_layers = 3
hidden = [2, 3, 2] # No. of nodes in each layer
# UPDATE THESE

num_epochs = 10000
print num_trainset
num_batches = num_trainset/10

out, net_in, net_in_bias, theta, error, dtheta, loss = initialize.init(X, num_layers, hidden, num_epochs)
#print theta[1]

for epoch in xrange(num_epochs):
	print epoch,
	for i in xrange(0, num_batches):

		if i == num_batches-2:
			out[0] = X[i*10:, :]
			temp_Y = Y[i*10:, :]
			
		else:
			out[0] = X[i*10:(i*10)+10, :]
			temp_Y = Y[i*10:i*10+10, :]
			
		#Forward Propagation
		out, net_in, net_in_bias = forwardprop.forward(num_layers, out[0].shape[0], net_in_bias, out, net_in, theta)
		
		#Backpropagation
		error, dtheta, theta = backprop.back(num_layers, out[0].shape[0], error, out, hidden, net_in_bias, theta, dtheta, alpha, temp_Y)
	
	out, _, _, _, _, _, _ = initialize.init(X, num_layers, hidden, num_batches)
	out, _, _ = forwardprop.forward(num_layers, num_trainset, net_in_bias, out, net_in, theta)		
	loss[epoch] = performance_metrics.loss(Y, out[num_layers-1], Y.shape[0])
	#plot.plot_boundary(X, Y_nb, out, num_layers, net_in_bias, net_in, theta)

out, _, _, _, _, _, _ = initialize.init(X, num_layers, hidden, num_batches)
out, _, _ = forwardprop.forward(num_layers, num_trainset, net_in_bias, out, net_in, theta)
print performance_metrics.confusion_matrix(Y, out[num_layers-1], num_class)
print "Accuracy: " + str(performance_metrics.accuracy(Y, out[num_layers-1]))
print "Precision: " + str(performance_metrics.precision(Y, out[num_layers-1], num_class))
print "Recall: " + str(performance_metrics.recall(Y, out[num_layers-1], num_class))
print "Error: " + str(performance_metrics.sum_error(error, num_layers))

plot.plot_boundary(X, Y_nb, out, num_layers, net_in_bias, net_in, theta)
plot.plot_loss(np.arange(num_epochs), loss)
