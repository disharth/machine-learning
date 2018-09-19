from keras.models import Sequential
from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.pipeline import Pipeline

from pandas import read_csv
import numpy
seed = 7
numpy.random.seed(seed)

dataframe = read_csv('sonar.csv' , header=None)
dataset = dataframe.values

X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

encoder = LabelEncoder()
encoder.fit(Y)

encoded_Y = encoder.transform(Y)
def baseline_model():
    model = Sequential()
    model.add(Dense(30, input_dim=60,kernel_initializer='normal',activation='relu'))
    model.add(Dense(1, kernel_initializer='normal',activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
estimators = []
estimators.append(('std', StandardScaler()))
optimizer = KerasClassifier(build_fn=baseline_model,epochs=100 , batch_size=5 , verbose=0)
estimators.append(('mlp',optimizer))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10,shuffle=True,random_state=7)
results =  cross_val_score(pipeline, X , encoded_Y ,cv=kfold)
print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))




