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


#Naive Bayes(Classification & Predict)
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
gnb_pred=gnb.predict(X_test)
print("True: ")
print(y_test)
print("GNB-Predict: ")
print(gnb_pred)
#GNB-CM
from sklearn.metrics import confusion_matrix
gnb_cm=confusion_matrix(y_test,gnb_pred)
print("GNB-CM: ")
print(gnb_cm)

