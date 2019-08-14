import pandas as pd
import numpy as np


data=pd.read_csv("Social_Network_Ads.csv")
x=data.iloc[:,2:4].values
y=data.iloc[:,4:].values


gender=data.iloc[:,1:2].values
purchased=data.iloc[:,4:].values



#Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
gender[:,0]=le.fit_transform(gender[:,0])

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features="all")
gender=ohe.fit_transform(gender).toarray()
print(gender)

#Data Joining
s1=pd.DataFrame(data=x,index=range(400),columns=["Age","EstimatedSalary"])
s2=pd.DataFrame(data=gender[:,0],index=range(400),columns=["Gender"])
s3=pd.DataFrame(data=purchased,index=range(400),columns=["Purchased"])
s4=pd.concat([s1,s2],axis=1)
newdata=pd.concat([s4,s3],axis=1) #x and y all in the newdata


X=newdata.iloc[:,0:3].values
Y=newdata.iloc[:,3:].values


#Dimension Reduction

#Train-Test-Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=0,test_size=0.33)

#Standard Scale
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(x_train)
X_test=ss.fit_transform(x_test)


#1-)PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train2=pca.fit_transform(X_train)
X_test2=pca.transform(X_test)

#LogisticRegression or anyone
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(random_state=0)
lr.fit(X_train2,y_train)
lr_pred=lr.predict(X_test2)

from sklearn.metrics import confusion_matrix
lr_cm=confusion_matrix(y_test,lr_pred)
print("PCA-CM: ")
print(lr_cm)


#2-)LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda=LDA(n_components=2)
X_train_lda=lda.fit_transform(X_train,y_train)
X_test_lda=lda.transform(X_test)

from sklearn.linear_model import LogisticRegression

lda_lr=LogisticRegression(random_state=0)
lda_lr.fit(X_train_lda,y_train)
lda_pred=lda_lr.predict(X_test_lda)

from sklearn.metrics import confusion_matrix
lda_cm=confusion_matrix(y_test,lda_pred)
print("LDA-CM: ")
print(lda_cm)




