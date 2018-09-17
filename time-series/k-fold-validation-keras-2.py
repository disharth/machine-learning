
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

import numpy

def create_model(optimizer='adam' , init='glorot_uniform'):
    model = Sequential()
    model.add(Dense(12,input_dim=8, kernel_initializer=init, activation='relu'))
    model.add(Dense(8 , activation='relu'))
    model.add(Dense(1 , activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

dataset  =  numpy.loadtxt('pima-indians-diabetes.csv' , delimiter=',')
X = dataset[:,0:8]
Y = dataset[:,8]
model = KerasClassifier(build_fn=create_model , epochs=150 , batch_size=10 , verbose=0)
inits = ['glorot_uniform', 'normal', 'uniform']
batches = [5,10,20]
epochs = [150, 50]
optimizers = ['adam' , 'rmsprop']
param_grid = dict(optimizer=optimizers , epochs = epochs , batch_size=batches, init=inits)
gs_cv = GridSearchCV(estimator=model , param_grid=param_grid)
grid_result = gs_cv.fit(X , Y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

