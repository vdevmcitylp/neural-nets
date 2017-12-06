import numpy as np
import read_data, label_binarize, performance_metrics, initialize, forwardprop, backprop

#np.random.seed(42)

alpha = 0.1

X, Y = read_data.input()

X = X.reshape(X.shape[0], X.shape[1])
Y = label_binarize.label_binarize(Y, classes = [0, 1])
#print Y

num_trainset = X.shape[0]
num_class = 2

# UPDATE THESE
num_layers = 6
hidden = [2, 3, 2, 2, 2, 2] # No. of nodes in each layer
#UPDATE THESE

num_epochs = 10

out, net_in, net_in_bias, theta, error, dtheta = initialize.init(X, num_layers, hidden)
#print theta[1]

for epoch in xrange(num_epochs):
	#print "Epoch: " + str(epoch),

	#Forward Propagation
	out, net_in, net_in_bias = forwardprop.forward(num_layers, num_trainset, net_in_bias, out, net_in, theta)
	
	#Backpropagation
	error, dtheta, theta = backprop.back(num_layers, num_trainset, error, out, hidden, net_in_bias, theta, dtheta, alpha, Y)
	#print error[3]	
	#print "Loss:" + str(performance_metrics.loss(Y, out[num_layers-1], num_trainset))

print performance_metrics.confusion_matrix(Y, out[num_layers-1], num_class)
print (performance_metrics.accuracy(Y, out[num_layers-1]))
# print "Precision: " + str(performance_metrics.precision(Y, out[num_layers-1], num_class))
# print "Recall: " + str(performance_metrics.recall(Y, out[num_layers-1], num_class))