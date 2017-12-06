import numpy as np
import label_binarize

def input():
	f = open('../../datasets/clusterincluster.txt', 'r')

	info = f.readlines()

	X = []
	Y = []

	for item in info:

		item = item.split()
		x1 = float(item[0])
		x2 = float(item[1])
		y = float(item[2])

		x = np.append(x1, x2)

		X.append(x)
		Y.append(y)

	X = np.array(X)
	Y = np.array(Y)

	return X, Y
		
# x = []
# y = []

# for row, col in enumerate(X):
# 	col = [float(i) for i in col]
# 	x.append(col)
 
# Y = map(int, Y)

# X = np.array(X)
# Y = np.array(Y)

# return X, Y
