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




            #Clustering

#1-)K-means algorithm
from sklearn.cluster import KMeans

kmeans=KMeans(n_clusters=3,init="k-means++")
kmeans.fit(X)
print("Cluster Centers: ")
print(kmeans.cluster_centers_)

kmdata=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=123)
    kmeans.fit(X)
    kmdata.append(kmeans.inertia_) #.inertia for wcss values,write and select(k)
plt.plot(range(1,11),kmdata)
plt.show()                         #and i select 3
                                   

#2-)Hierarchical (I like this clustering.) 

#Find the best k values with Dendogram
import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(X,method="ward"))
plt.show()
#okay i select k values=3   but 2 better

#Hierarchical
from sklearn.cluster import AgglomerativeClustering
AC=AgglomerativeClustering(n_clusters=3,linkage="ward",affinity="euclidean")
AC_pred=AC.fit_predict(X)
plt.scatter(X[AC_pred==0,0],X[AC_pred==0,1],s=50,color="red")
plt.scatter(X[AC_pred==1,0],X[AC_pred==1,1],s=50,color="blue")
plt.scatter(X[AC_pred==2,0],X[AC_pred==2,1],s=50,color="green")





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
print(layers_pred)
from sklearn.metrics import confusion_matrix
layers_cm=confusion_matrix(y_test,layers_pred)
print(layers_cm)















