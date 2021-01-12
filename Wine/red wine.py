# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:06:51 2020

@author: cvveljanovski
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from patsy import dmatrices

data = pd.read_csv("wineQualityReds.csv")

check = data.describe()
#print(check)
"""
plt.rcParams['figure.figsize'] = (10,5)
fig, axes = plt.subplots(nrows = 2, ncols = 2)

num_features = ["citric.acid", "sulphates", "alcohol", "quality"]
xaxes = num_features
yaxes = ["Counts","Counts","Counts","Counts"]

axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.hist(data[num_features[idx]].dropna(), bins=20)
    ax.set_xlabel(xaxes[idx], fontsize=15)
    ax.set_ylabel(yaxes[idx], fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
"""
"""   
plt.matshow(data.corr())
plt.xticks(range(len(data.columns)),data.columns)
plt.yticks(range(len(data.columns)),data.columns)
plt.colorbar()
plt.show()
"""
"""
#See the skewnes of the data
plt.figure(figsize=(20,16))
for i, col in enumerate(list(data.columns.values)):
    plt.subplot(4, 3, i+1)
    sns.distplot(data[col], color = 'r', kde = True, label = 'data') 
    plt.grid()
    plt.legend(loc='upper_right')
    plt.tight_layout()
"""

condition = [
        (data['quality'] >= 7),
        (data['quality'] < 5)
        ]
rating = ["very good", "bad"]
data['rating'] = np.select(condition, rating, default = 'good')
"""
check1 = data.rating.value_counts()
print(check1)
"""
check2 = data.groupby('rating').mean()
#print(check2)

correlation = data.corr()
correl = correlation['quality'].sort_values(ascending=False)
#print(correl)
#We see that alcohol, sulphates and citric.acid are the clear decidision makers

#sulphates = sns.boxplot(x='rating', y='sulphates', data=data)
#alcohol = sns.boxplot(x='rating', y='alcohol', data=data)
#citric = sns.boxplot(x='rating', y='citric.acid', data=data)
#This means that high citric.acid means very good wine

#Add a feature rating that categorizes the wines as good 1 and bad 0
data['rating'] = (data['quality'] > 5).astype(np.float32)
data['citric'] = (data['citric.acid']).astype(np.float32)

y1,X1 = dmatrices('rating ~ citric', data=data)
sns.distplot(X1[y1[:,0]>0,1], color = "blue")
sns.distplot(X1[y1[:,0] == 0, 1], color = "yellow")

y,X = dmatrices('rating ~ sulphates', data=data)
sns.distplot(X[y[:,0]>0,1], color = "green")
sns.distplot(X[y[:,0] == 0, 1], color = "red")

#From this graph we can see that the good wines have higher sulphates value as well as higher alcohol value
