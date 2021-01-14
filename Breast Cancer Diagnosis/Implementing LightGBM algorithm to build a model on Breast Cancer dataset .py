#!/usr/bin/env python
# coding: utf-8

# In[23]:


# In this notebook we will see how we can implement the Boosting algorithm LightGBM and as an example we will use the "Breast Cancer Diagnosis dataset".


# In[24]:


# Standard frameworks
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Model Selection frameworks
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_validate

# Import the algorithm for the model
import lightgbm as lgb

# Accuracy measurements
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[25]:


data = pd.read_csv(r"C:\Users\cvveljanovski\Desktop\Learning\DataScience\Datasets\Breast_cancer_data.csv")
data.head()


# # Splitting the data in training and testing

# In[26]:


Y = data.diagnosis
X = data.drop("diagnosis",axis=1)


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# # Creating the model with LightGBM

# In[28]:


model = lgb.LGBMClassifier()
model.fit(X_train, y_train)


# # Do prediction with the model

# In[29]:


pred_train = model.predict(X_train)
pred = model.predict(X_test)


# # Check the accuracy of the model

# In[30]:


Accuracy_train = accuracy_score(pred_train, y_train)
Accuracy = accuracy_score(pred, y_test)


# In[31]:


print("The accracy score on the training dataset of the model created with the LightGBM algorithm is: {0:0.4f}".format(accuracy_score(y_train, pred_train)))
print("The accracy score of the model created with the LightGBM algorithm is: {0:0.4f}".format(accuracy_score(y_test, pred)))


# ### As we can see the accuracy of the model on the training dataset is 1 and on the testing 0.9298 which is understandable
# ### Since the model was trained on the training dataset

# # We can represent the performance of the model with a confusion matrix

# In[32]:


print(classification_report(y_test, pred))

