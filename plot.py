import numpy as np
import forwardprop, performance_metrics
import matplotlib.pyplot as plt

def plot_boundary(X, Y_nb, out, num_layers, net_in_bias, net_in, theta, epoch, Z, xx, yy):

	# h = 0.02
	# x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
	# y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1

	# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	
	# Z = np.c_[xx.ravel(), yy.ravel()]
	out[0] = Z
	
	prediction, y, z = forwardprop.forward(num_layers, Z.shape[0], net_in_bias, out, net_in, theta)
	pred = np.argmax(prediction[num_layers-1], axis = 1)
	pred = pred.reshape(xx.shape)
	plt.contourf(xx, yy, pred, cmap = plt.cm.Paired, alpha = 0.8)
	plt.scatter(X[:, 0], X[:, 1], c = Y_nb, cmap = plt.cm.Paired)
	plt.xlabel('x_1')
	plt.ylabel('x_2')
	plt.title('Decision Boundary')
	#plt.show()
	plt.savefig('../Photos/img' + str(epoch) + '.png')

def plot_loss(epoch_no, loss):

	plt.plot(epoch_no, loss)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')
	plt.title('Loss')
	plt.show()