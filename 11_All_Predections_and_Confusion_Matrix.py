import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("Social_Network_Ads.csv")
x=data.iloc[:,2:4]
y=data.iloc[:,4:]
X=x.values
Y=y.values
gender=data.iloc[:,1:2].values
purchased=data.iloc[:,4:].values

#Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
gender[:,0]=le.fit_transform(gender[:,0])

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features="all")
gender=ohe.fit_transform(gender).toarray()


#Data Joining
s1=pd.DataFrame(data=x,index=range(400),columns=["Age","EstimatedSalary"])
s2=pd.DataFrame(data=purchased,index=range(400),columns=["Purchased"])
"""
gender[:,0] for Dummy Variable
"""
join=pd.concat([s1,s2],axis=1)


#TrainTest Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=0,test_size=0.33)

#Linear Regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
lr_pred=lr.predict(x_test)
lr_pred[lr_pred >= 0.5]=1
lr_pred[lr_pred < 0.5]=0
print("True: ")
print(y_test)
print("LR-Predict: ")
print(lr_pred)


#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
pf=PolynomialFeatures(degree=4)
x_pf=pf.fit_transform(X)
lr2=LinearRegression()
lr2.fit(x_train,y_train)
poly_pred=lr2.predict(x_test)
poly_pred[poly_pred >= 0.5]=1
poly_pred[poly_pred<0.5]=0
print("True: ")
print(y_test)
print("Poly-Predict: ")
print(poly_pred)

"""
Polynomial Regression need to Linear Regression
"""

#Support Vector Regression
from sklearn.svm import SVR
svr=SVR(kernel="rbf")
svr.fit(x_train,y_train)
svr_pred=svr.predict(x_test)
svr_pred[svr_pred>=0.5]=1
svr_pred[svr_pred<0.5]=0
print("True: ")
print(y_test)
print("SVR-Predict: ")
print(svr_pred)

#Decision  Tree Regression
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=0)
dtr.fit(x_train,y_train)
dtr_pred=dtr.predict(x_test)
print("True: ")
print(y_test)
print("DTR-Predict: ")
print(dtr_pred)

#Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=400,random_state=0)
rfr.fit(x_train,y_train)
rfr_pred=rfr.predict(x_test)
rfr_pred[rfr_pred>=0.5]=1
rfr_pred[rfr_pred<0.5]=0
print("True: ")
print(y_test)
print("RFR-Predict: ")
print(rfr_pred)

#Confusion Matrix

"""
how much prediction is correct left-up and right-down is correct prediction number
exempla:lr_cm>>> LR-CM:
                [[78  6]
                [12 36]]
        correct=78+36,wrong=12+6
"""

from sklearn.metrics import confusion_matrix
lr_cm=confusion_matrix(y_test,lr_pred)
print("LR-CM: ")
print(lr_cm)

poly_cm=confusion_matrix(y_test,poly_pred)
print("POLY-CM: ")
print(poly_cm)

svr_cm=confusion_matrix(y_test,svr_pred)
print("SVR-CM: ")
print(svr_cm)

dtr_cm=confusion_matrix(y_test,dtr_pred)
print("DTR-CM: ")
print(dtr_cm)

rfr_cm=confusion_matrix(y_test,rfr_pred)
print("RFR-CM: ")
print(rfr_cm)









