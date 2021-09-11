# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import pandas as pd
#Importing Dataset 
b_dataset = pd.read_csv("updatedgradedata.csv")


#Converting into x(data) and y_normal (predictions)
x  = b_dataset.iloc[:,1:-1]
y_normal  = b_dataset.iloc[:,-1]


##############################################################################
#PreProcessing
##############################################################################
#Convrting the y_normal predictions into 0 to 10 form 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
lb_en_y = LabelEncoder()
y = lb_en_y.fit_transform(y_normal)
#One Hot Encoding
hotencoder = OneHotEncoder()
y=y.reshape(-1, 1)
y = hotencoder.fit_transform(y).toarray()

#Split dataset into traing (85%) and test(15%) set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15)

#Scaling the values of x_train 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

##############################################################################
#Neural Network
##############################################################################

#Importing the Dataset
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

#Neural Network Model
model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(1000, activation='relu', input_dim=14))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(250, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(9, activation='softmax'))

model = keras.models.load_model("GradePrediction85.HDF5")
#momentum: float hyperparameter >= 0 that accelerates gradient descent in the relevant direction and dampens oscillations. Defaults to 0, i.e., vanilla gradient descent.
#nesterov: boolean. Whether to apply Nesterov momentum. Defaults to False.
"""
##############################################################################
#Training
##############################################################################
model.compile(optimizer=Adam(lr=0.0001), loss="categorical_crossentropy",metrics=["accuracy"])
model.fit(x_train, y_train,
epochs=300,
batch_size=18)
score = model.evaluate(x_test, y_test, batch_size=18)
model.save("GradePrediction.HDF5")
"""
correct = 0
#Predicting x_test
for i in range(len(x_test)):
    aa = np.expand_dims(x_test[i], axis=0)
    pred = model.predict(aa)
    index=np.where(pred==pred.max())
    pred_values=lb_en_y.inverse_transform(index[1])[0]
    actual_values= lb_en_y.inverse_transform(np.where(y_test[i]==1)[0])[0]
    print("Pred ",pred_values)
    print("Actual ",actual_values)
    print("#########")
    if pred_values == actual_values:
        correct+=1
        print(correct)
print(correct/len(y_test))



def predict(aa):
    aa = np.expand_dims(aa, axis=0)
    pred = model.predict(aa)
    index=np.where(pred==pred.max())
    return lb_en_y.inverse_transform(index[1])[0]
    












































































