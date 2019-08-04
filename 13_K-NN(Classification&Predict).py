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


#K-NN(Classification & Predict),KNN=K Nearest Neighborhood
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3,metric="minkowski")
knn.fit(X_train,y_train)
knn_pred=knn.predict(X_test)
print("True: ")
print(y_test)
print("KNN-Predict: ")
print(knn_pred)
#KNN-CM
from sklearn.metrics import confusion_matrix
knn_cm=confusion_matrix(y_test,knn_pred)
print("KNN-CM: ")
print(knn_cm)







