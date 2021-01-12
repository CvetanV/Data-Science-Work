# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 16:06:51 2020

@author: cvveljanovski
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from patsy import dmatrices

data = pd.read_csv("wineQualityWhites.csv")
#check1= data.describe()
#print(check1)

#Normal graph from matplotlob
"""
plt.rcParams['figure.figsize']=(10,5)
fig,axes = plt.subplots(nrows = 2, ncols = 3)
num_features = ['fixed.acidity', 'density', 'pH', 'alcohol','quality']

xaxes = num_features
yaxes = ["Counts", "Counts", "Counts", "Counts", "Counts"]
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.hist(data[num_features[idx]].dropna(), bins=20)
    ax.set_xlabel(xaxes[idx], fontsize=15)
    ax.set_ylabel(yaxes[idx], fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
"""
"""
for col in data.columns.values:
    print("Number of unique values of {} : {}".format(col, data[col].nunique()))
"""
"""
#Seaborn graph    
sns.catplot(x="quality", data = data, kind = "count")
"""
"""
#Correlation heatmap graph amont the different featuers in seaborn 
plt.figure(figsize=(10, 10))
sns.heatmap(data.corr(), color = "k", annot = True)
"""
"""
plt.figure(figsize = (10, 15))
for i, col in enumerate(list(data.columns.values)):
    plt.subplot(4, 3, i+1)
    data.boxplot(col)
    plt.grid()
    plt.tight_layout()
   """ 
   
#Check the distribution skewnes
"""
plt.figure(figsize=(20,16))
for i, col in enumerate(list(data.columns.values)):
    plt.subplot(4, 3, i+1)
    sns.distplot(data[col], color = 'r', kde = True, label = 'data') 
    plt.grid()
    plt.legend(loc='upper_right')
    plt.tight_layout()
"""

#How many good and bad black wines do we have in the dataset
conditions = [
        (data['quality'] >= 7),
        (data['quality'] <= 4)
        ]
rating = ['very good', 'bad']
data['rating'] = np.select(conditions, rating, default = 'good')
check2 = data.rating.value_counts()
#print(check2)

#Based on the ratings what are the values for the different features for the wines on each rating 
check2 = data.groupby('rating').mean()
#print(check2)
"""We see that good wines have low "volatile acidity", low "value of chlorides", low "total sulfur dioxides", high "value of sulphates"
"""

"""
correlation = data.corr()       
correl = correlation['quality'].sort_values(ascending=False)
#print(correl) 

chlorides = sns.boxplot(x = 'rating', y = 'total.sulfur.dioxide', data = data)
chlorides.set(xlabel = 'Wine ratings', ylabel = 'Chlorides in wine', title = 'Chlorides in wine per rating') 
"""

#Lets sellect out features(we can use residual sugar, total sulfur dioxides, clorides or volatile acidity)
"""sns.lmplot(x='alcohol', y='residual.sugar', col='rating', data=data)"""
#This means that for a good wine it is important to have more residual sugars and higher alcohol content

#Let's make another binary feature rate_code that will classify the wines as 1 or 0 if they have quality > 5 or not 
data['rate_code'] = (data['quality'] > 5).astype(np.float32)

y,X = dmatrices('rate_code ~ alcohol', data=data)
sns.distplot(X[y[:,0]>0,1], color = "green")
sns.distplot(X[y[:,0] == 0, 1], color = "red")