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

#Decision Tree(Classification & Predict)
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train,y_train)
dtc_pred=dtc.predict(X_test)
print("True: ")
print(y_test)
print("DTC-Predict: ")
print(dtc_pred)
#DTC-CM
from sklearn.metrics import confusion_matrix
dtc_cm=confusion_matrix(y_test,dtc_pred)
print("DTC-CM: ")
print(dtc_cm)












