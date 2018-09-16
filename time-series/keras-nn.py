# Keras imports
from keras.models import Sequential
from keras.layers import Dense

# The numpy imports
import numpy

dataset = numpy.loadtxt('pima-indians-diabetes.csv' , delimiter=',')

# input variables
X = dataset[:,0:8]

# output variable
Y = dataset[:,8]

# create network architecture
model = Sequential()
model.add(Dense(12 , input_dim=8,activation='relu'))
model.add(Dense(8 , activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# create computation graph
model.compile(loss='binary_crossentropy' , optimizer='adam',metrics=['accuracy'])

# run the graph
model.fit(X,Y,epochs=150, batch_size=10)

# evaluate the model on training set
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


