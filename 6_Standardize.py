import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("Social_Network_Ads.csv")
x=data.iloc[:,2:4]
y=data.iloc[:,4:]
X=x.values
Y=y.values
gender=data.iloc[:,1:2].values

#Encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
gender[:,0]=le.fit_transform(gender[:,0])

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(categorical_features="all")
gender=ohe.fit_transform(gender).toarray()


#Data Joining
s1=pd.DataFrame(data=x,index=range(400),columns=["Age","EstimatedSalary"])
s2=pd.DataFrame(data=gender[:,0],index=range(400),columns=["Gender"])
join=pd.concat([s1,s2],axis=1)

#TrainTest Split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=0,test_size=0.33)

#Standardize
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X_train=ss.fit_transform(x_train)
X_test=ss.fit_transform(x_test)



































