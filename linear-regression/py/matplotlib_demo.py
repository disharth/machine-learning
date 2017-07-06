from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
data = np.genfromtxt ('ex1data2.txt', delimiter=",")
num_features = data.shape[1]
num_records = data.shape[0]
ITERATIONS = 700
alpha = 0.1
normalize = True # Choose if normalization needed.
X = np.array(data[:,0:num_features -1])
Y = np.array(data[:,(num_features - 1):num_features])
y = Y.reshape(num_records,)
meanX = X.mean(axis=0)
stdX = X.std(axis=0)
if normalize:
    X = (X - meanX )/stdX

fig = plt.figure()
ax1 = fig.add_subplot(111,projection='3d')
ax1.scatter(X[:,0],X[:,1],y)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

plt.show()