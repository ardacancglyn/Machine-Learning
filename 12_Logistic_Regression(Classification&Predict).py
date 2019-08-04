import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("Social_Network_Ads.csv")
x=data.iloc[:,2:4]
y=data.iloc[:,4:]
X=x.values
Y=y.values

#Classification!!!

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=0,test_size=0.33)


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(x_train)
X_test=ss.fit_transform(x_test)
#Logistic Regression(Classification & Predict)
from sklearn.linear_model import LogisticRegression
lgr=LogisticRegression(random_state=0)
lgr.fit(X_train,y_train)
lgr_pred=lgr.predict(X_test)
print("Pred: ")
print(lgr_pred)
#LGR-CM
from sklearn.metrics import confusion_matrix
lgr_cm=confusion_matrix(y_test,lgr_pred)
print("True: ")
print(y_test)
print("LGR-CM: ")
print(lgr_cm)

















