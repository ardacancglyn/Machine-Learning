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


#Backward Elimination
#What the B.E: delete unnecessary variable  

import statsmodels.api as sm
#Step 1-) Append One(1)
appendones=np.append(arr=np.ones((400,1)).astype(int),values=newdata,axis=1)

#Step 2-) İf P>t values >0.50  delete column.
elimination1=newdata.iloc[:,[0,1,2,3]].values

elimination2=sm.OLS(endog=purchased,exog=elimination1).fit()
print(elimination2.summary())

#   2nd column  P>t values >0.50 and delete
elimination1=newdata.iloc[:,[0,1,3]].values

elimination2=sm.OLS(endog=purchased,exog=elimination1).fit()
print(elimination2.summary())

X=newdata.iloc[:,0:2].values  # new ndependent variable
Y=newdata.iloc[:,3:].values   # new depented variable








