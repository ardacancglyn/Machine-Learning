import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("Social_Network_Ads.csv")
x=data.iloc[:,2:4]
y=data.iloc[:,4:]
X=x.values
Y=y.values

"""
user=data[["User ID"]]
print(user)

gender=data[["Gender"]]
printint(gender)

"""



