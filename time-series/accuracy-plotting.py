from keras.models import Sequential
from keras.layers import Dense

import matplotlib.pyplot as plt

import numpy

from sklearn.model_selection import StratifiedKFold

dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')

X = dataset[:,0:8]
Y = dataset[:,8]
model = Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy' , optimizer='adam',metrics=['accuracy'])
history = model.fit(X , Y, validation_split=0.33, epochs=150 , batch_size=10,verbose=0)

print(history)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# summarize history for loss 
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left') 
plt.show()
