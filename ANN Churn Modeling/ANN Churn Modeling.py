#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import basic frameworks
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Feature Scaling
from sklearn.preprocessing import StandardScaler

#Import model selection
from sklearn import model_selection
from sklearn.model_selection import train_test_split,cross_validate

#ANN frameworks
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, PReLU, ELU

#Import measurement metrics
from sklearn.metrics import confusion_matrix


# In[2]:


#Read CSV file
data = pd.read_csv(r"C:\\Users\\cvveljanovski\\Desktop\\Learning\\DataScience\\Datasets\\ANN\\Churn_Modelling.csv")


# In[3]:


#We are going to drop the columns that we don't need
data.drop(["RowNumber","CustomerId", "Surname"],inplace=True,axis=1) 


# In[4]:


data.head()


# # EDA

# In[5]:


# Distribution of customers by Geography
sns.factorplot(y="Geography", data = data, kind = "count", size = 2, aspect = 3)


# #### Many customers are from France

# In[6]:


# Distribution of customers by Gender
sns.factorplot(y="Gender", data = data, kind = "count", size = 2, aspect = 3)


# In[7]:


# Distribution of customers by Age
sns.factorplot(y="Age", data = data, kind = "count", size = 10, aspect = 1.5)


# In[8]:


sns.boxplot(data=data[["Age"]])
fig=plt.gcf()
fig.set_size_inches(5,5)


# #### As expected we have the most number of customers between 27 and 46

# In[9]:


# Distribution of customers by Tenure
sns.factorplot(y="Tenure", data = data, kind = "count", size = 3, aspect = 2)


# In[10]:


# Distribution of customers by Num of product
sns.factorplot(y="NumOfProducts", data = data, kind = "count", size = 2, aspect = 3)


# #### We can see that the customers use mostly 1 or 2 product from the bank

# In[11]:


# Distribution of customers by Has Credit card
sns.factorplot(y="HasCrCard", data = data, kind = "count", size = 2, aspect = 3)


# In[12]:


# Distribution of customers by Is Active Member
sns.factorplot(y="IsActiveMember", data = data, kind = "count", size = 2, aspect = 3)


# In[13]:


# Distribution of customers by Exited the bank
sns.factorplot(y="Exited", data = data, kind = "count", size = 2, aspect = 3)


# In[14]:


# Lets see the correlation among the different features
cor_mat=data[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)]=False
fig=plt.gcf()
fig.set_size_inches(10,10)
sns.heatmap(data=cor_mat, mask=mask, square=True, annot=True, cbar=True)


# # Feature engineering

# In[16]:


#Which are the categorical features
cat = data.select_dtypes(include=["object"])
cat.head()


# In[17]:


# One hot encoding on the categorical features
cat_encoded = pd.get_dummies(cat)


# In[18]:


# Now lets collect the data in one dataframe
df=pd.concat([data,cat_encoded],axis=1)
# Drop the features that we have performed one hot encoding on
df.drop(["Geography","Gender"],inplace=True,axis=1)
df.head()


# # Now we will split the original dataframe in dataframe with independent features and df with dependent feature

# In[19]:


# Goal is to predict if a customer will exit the bank or not
Y = df.Exited
Y.head()


# In[20]:


X = df.drop("Exited",axis=1)
X.head()


# # Split the data in train and test

# In[21]:


X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=0)


# In[22]:


[X_train.shape, X_test.shape, Y_train.shape, Y_test.shape]


# In[23]:


#Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[40]:


# Initialising the ANN
classifier = Sequential()


# In[54]:


# Adding the input layer and the first hidden layer
classifier.add(Dense(6, activation='relu', kernel_initializer='he_uniform', input_dim=13))
# We can add also dropout to prevent overfitting
classifier.add(Dropout(0.1))


# In[55]:


# Adding the second hidden layer
classifier.add(Dense(6, activation='relu', kernel_initializer='he_uniform'))
# Adding dropout to prevent overfitting
classifier.add(Dropout(0.1))


# In[56]:


# Adding the output layer (output_dim is 1 as we want only 1 output from the final layer.)
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='he_uniform'))


# In[57]:


# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[58]:


# Fitting the ANN to the Training set
model_history = classifier.fit(X_train, Y_train, validation_split = 0.33, batch_size=10, epochs=100)


# In[59]:


print(model_history.history.keys())


# In[62]:


plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend(['train','test'], loc = 'upper left')
plt.show()


# In[ ]:




