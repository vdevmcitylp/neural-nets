import numpy as np

def plot(X):

	h = 0.02
	x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
	y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1

	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	
	return xx, yy