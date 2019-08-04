import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("Social_Network_Ads.csv")
x=data.iloc[:,2:4]
y=data.iloc[:,4:]
X=x.values
Y=y.values


            #Clustering

#1-)K-means algortihm
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











