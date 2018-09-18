import numpy
from pandas import read_csv

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

dataframe = read_csv('iris.csv' , header=None)

dataset = dataframe.values
X = dataset[:,0:4]
Y =  dataset[:,4]

encoder = LabelEncoder()
encoder.fit(Y)
encoder_Y = encoder.transform(Y)

dummy_y = np_utils.to_categorical(encoder_Y)

def baseline_model():
    model = Sequential()
    model.add(Dense(8, input_dim=4 , activation='relu'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy' ,optimizer='adam' , metrics=['accuracy'])
    return model

estimator = KerasClassifier(build_fn=baseline_model , epochs=200 , batch_size=5 , verbose=0)

kfold = KFold(n_splits=10 , shuffle=True , random_state=7)

results = cross_val_score(estimator, X, dummy_y, cv=kfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))






