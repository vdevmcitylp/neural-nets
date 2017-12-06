import numpy as np
import read_data, label_binarize, performance_metrics, initialize, forwardprop, backprop, plot
import matplotlib.pyplot as plt

#np.random.seed(42)

alpha = 0.3

X, Y = read_data.input()

X = X.reshape(X.shape[0], X.shape[1])
Y = label_binarize.label_binarize(Y, classes = [0, 1])
#print Y
#print X.shape
num_trainset = X.shape[0]

# UPDATE THESE
num_class = 2
num_layers = 3
hidden = [2, 6, 2] # No. of nodes in each layer
# UPDATE THESE

num_epochs = 20

out, net_in, net_in_bias, theta, error, dtheta = initialize.init(X, num_layers, hidden)
#print theta[1]

xx, yy = plot.plot(X)

for epoch in xrange(num_epochs):
	#print "Epoch: " + str(epoch),

	#Forward Propagation
	out, net_in, net_in_bias = forwardprop.forward(num_layers, num_trainset, net_in_bias, out, net_in, theta)
	
	#Backpropagation
	error, dtheta, theta = backprop.back(num_layers, num_trainset, error, out, hidden, net_in_bias, theta, dtheta, alpha, Y)
	#print error[3]	
	#print "Loss:" + str(performance_metrics.loss(Y, out[num_layers-1], num_trainset))

Z = np.c_[xx.ravel(), yy.ravel()]
out[0] = Z
prediction, y, z = forwardprop.forward(num_layers, 568964, net_in_bias, out, net_in, theta)
pred = np.argmax(prediction[num_layers-1], axis = 1)
pred = pred.reshape(xx.shape)
plt.contourf(xx, yy, pred, cmap=plt.cm.Paired, alpha=0.8)
plt.show()
# print performance_metrics.confusion_matrix(Y, out[num_layers-1], num_class)
#print (performance_metrics.accuracy(Y, out[num_layers-1]))
# print "Precision: " + str(performance_metrics.precision(Y, out[num_layers-1], num_class))
# print "Recall: " + str(performance_metrics.recall(Y, out[num_layers-1], num_class))