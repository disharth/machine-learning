from keras.models import Sequential
from keras.layers import Dense

import numpy

from sklearn.model_selection import StratifiedKFold

dataset = numpy.loadtxt('pima-indians-diabetes.csv', delimiter=',')

X = dataset[:,0:8]
Y = dataset[:,8]

kfold = StratifiedKFold(n_splits=10 , shuffle=True , random_state=7)

cvscores = []

for train , test in kfold.split(X,Y):
    model = Sequential()
    model.add(Dense(12,input_dim=8,activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy' , optimizer='adam',metrics=['accuracy'])
    model.fit(X[train] , Y[train], epochs=150 , batch_size=10,verbose=0)
    scores = model.evaluate(X[test], Y[test], verbose=0)
    print('%s: %.2f%%' % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1]*100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))