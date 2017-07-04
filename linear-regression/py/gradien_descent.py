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
x = np.insert(X,0,1,axis=1)
theta = np.zeros(num_features)
iterations = []
costs = []
for i in range(0,ITERATIONS):
    new_theta = np.zeros(num_features)
    diff = np.array(x.dot(theta) -y)
    diffX = diff.dot(x)
    for feature in range(0,num_features):
        new_theta[feature] = theta[feature] - (alpha/num_records)*diffX[feature]
    theta = new_theta
    iterations.append(i)
    costs.append((np.sum(np.square(diff))) / (2 * num_records))

# Plot the iterations vs cost. It should decrease on every iteration.
plt.plot(iterations, costs)
plt.show()
print(theta)  # This value of the theta can be used for predictions.
# Creating normalize data and then calculate hypothesis which is prediction.
d = np.array([1,(1650 - meanX[0])/stdX[0] , (3 - meanX[1])/stdX[1]])
print(d.dot(theta))

