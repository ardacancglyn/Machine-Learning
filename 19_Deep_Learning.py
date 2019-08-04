import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("Social_Network_Ads.csv")
x=data.iloc[:,2:4]
y=data.iloc[:,4:]
X=x.values
Y=y.values
                          #Classification!!!
#Train-Test-Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=0,test_size=0.33)

#Scale
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(x_train)
X_test=ss.fit_transform(x_test)


                        #Deep Learning
import keras
from keras.models import Sequential
#Neural Network
from keras.layers import Dense 
#Neuron and Layer 

classifier=Sequential()

    #Layers
#Start L.
classifier.add(Dense(10,init="uniform",activation="relu",input_dim=2))
#HiddenL.
classifier.add(Dense(8,init="uniform",activation="relu"))
classifier.add(Dense(6,init="uniform",activation="relu"))
classifier.add(Dense(4,init="uniform",activation="relu"))
classifier.add(Dense(2,init="uniform",activation="relu"))
#Finish L.
classifier.add(Dense(1,init="uniform",activation="sigmoid"))

classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
#Accuracy Compile

classifier.fit(X_train,y_train,epochs=100)#epochs = "random"  your choice
layers_pred=classifier.predict(X_test)
layers_pred = (layers_pred > 0.5) #True=1,False=0
print("True: ")
print(y_test)
print("Layers-Pred: ")
print(layers_pred
from sklearn.metrics import confusion_matrix
layers_cm=confusion_matrix(y_test,layers_pred)
print(layers_cm)
















