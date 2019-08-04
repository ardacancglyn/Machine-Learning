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

#SVM(Classification & Predict),SVM=Support Vector Machine
from sklearn.svm import SVC

#rbf
svc_rbf=SVC(kernel="rbf")
svc_rbf.fit(X_train,y_train)
svc_rbf_pred=svc_rbf.predict(X_test)
print("True: ")
print(y_test)
print("SVC-RBF-Predict: ")
print(svc_rbf_pred)

#poly
svc_poly=SVC(kernel="poly")
svc_poly.fit(X_train,y_train)
svc_poly_pred=svc_poly.predict(X_test)
print("True: ")
print(y_test)
print("SVC-POLY-Predict: ")
print(svc_poly_pred)

#linear
svc_linear=SVC(kernel="linear")
svc_linear.fit(X_train,y_train)
svc_linear_pred=svc_linear.predict(X_test)
print("True: ")
print(y_test)
print("SVC-LİNEAR-Predict: ")
print(svc_linear_pred)

#sigmoid
svc_sigmoid=SVC(kernel="sigmoid")
svc_sigmoid.fit(X_train,y_train)
svc_sigmoid_pred=svc_sigmoid.predict(X_test)
print("True: ")
print(y_test)
print("SVC-SİGMOİD-Predict: ")
print(svc_sigmoid_pred)



#SVM-CM
from sklearn.metrics import confusion_matrix
svc_rbf_cm=confusion_matrix(y_test,svc_rbf_pred)
print("SVC-RBF-CM: ")
print(svc_rbf_cm)

svc_poly_cm=confusion_matrix(y_test,svc_poly_pred)
print("SVC-POLY-CM: ")
print(svc_poly_cm)

svc_linear_cm=confusion_matrix(y_test,svc_linear_pred)
print("SVC-LİNEAR-CM: ")
print(svc_linear_cm)

svc_sigmoid_cm=confusion_matrix(y_test,svc_sigmoid_pred)
print("SVC-SİGMOİD-CM: ")
print(svc_sigmoid_cm)






