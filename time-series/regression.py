from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline

from pandas import read_csv
import numpy


dataframe = read_csv('housing.csv' , delim_whitespace=True, header=None)
dataset = dataframe.values

X = dataset[:,0:13]
Y = dataset[:,13]

def baseline_model():
    model = Sequential()
    model.add(Dense(20, input_dim=13,kernel_initializer='normal',activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
seed = 7
numpy.random.seed(seed)
estimators = []
estimators.append(('std', StandardScaler()))
optimizer = KerasRegressor(build_fn=baseline_model,epochs=100 , batch_size=5 , verbose=0)
estimators.append(('mlp',optimizer))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10,shuffle=True,random_state=7)
results =  cross_val_score(pipeline, X , Y ,cv=kfold)
print(results)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))




