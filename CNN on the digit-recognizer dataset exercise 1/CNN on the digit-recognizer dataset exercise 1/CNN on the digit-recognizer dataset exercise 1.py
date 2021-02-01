#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Feature Scaling
from sklearn.preprocessing import StandardScaler

#Import model selection
from sklearn import model_selection
from sklearn.model_selection import train_test_split,cross_validate

# CNN frameworks
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, PReLU, ELU

#Import measurement metrics
from sklearn.metrics import confusion_matrix, classification_report


# In[4]:


train_df= pd.read_csv(r'C:\Users\Legion Y530\Desktop\digit-recognizer\train.csv')
test_df= pd.read_csv(r'C:\Users\Legion Y530\Desktop\digit-recognizer\test.csv')


# # Data understanding

# In[5]:


train_df.head()


# In[6]:


test_df.head()


# In[15]:


# Check the value of the label for the record on row 40059 column 0
train_df.iloc[40059,0]


# In[10]:


# Take a random sample of 5 records
train_df.sample(5)


# In[23]:


# converting the training dataframe to numpy array to so that since each row of the dataframe represents a character
# the row will be an array of dim 28x28 and it will be stored in to an array of arrays
train_df.iloc[4,1:].to_numpy().reshape(28,28)


# In[24]:


# Lets see which character is on line 40059 by taking the row from the dataframe and reshaping it into 28x28 dim
plt.imshow(train_df.iloc[40059,1:].to_numpy().reshape(28,28)) 


# In[25]:


# digit visualization
plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(5,5,i+1) # creating subplot
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_df.iloc[i,1:].to_numpy().reshape(28,28))
    plt.xlabel([train_df.iloc[i,0]])
plt.show()


# # Data splitting

# In[27]:


X= train_df.drop(['label'], axis=1)
y= train_df['label']


# In[45]:


X_train, X_val, y_train, y_val = train_test_split(X,y, test_size = 0.2, random_state=0)


# In[46]:


X_train.head()


# In[47]:


X_train= X_train/255
X_val= X_val/255


# In[48]:


X_train.shape


# In[49]:


X_train = X_train.to_numpy().reshape(-1,28,28,1)
X_train.shape


# In[50]:


X_val = X_val.to_numpy().reshape(-1,28,28,1)
X_val.shape


# # Creating the CNN model

# In[53]:


model = keras.Sequential([
    #cnn
    layers.Conv2D(56, (3,3), padding='same',activation='relu', input_shape=(28,28, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(28, (3,3),padding='same', activation='relu', input_shape=(28,128, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(14, (3,3),padding='same', activation='relu', input_shape=(28,128, 1)),
    layers.MaxPooling2D((2,2)),
    
    #dense
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])
model.summary()


# In[54]:


# Let's compile the model
model.compile(
     optimizer='adam',
     loss='sparse_categorical_crossentropy',
     metrics=["accuracy"]
)
history= model.fit(X_train, y_train, epochs=50)


# In[55]:


# Prediction and Evaluation
model.evaluate(X_val,y_val)


# In[56]:


pred= model.predict(X_val)


# In[57]:


pred= np.argmax(pred, axis=1)
pred[:10]


# # Classification report & Confusion Matrix

# In[62]:


print(classification_report(y_val, pred))


# In[66]:


plt.figure(figsize=(8,6))
plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['loss'], label='loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid('on')  


# # On the test dataset

# In[67]:


# Reshape the dataset into an array of arrays with dim 28x28 
test_df=test_df.to_numpy().reshape(-1,28,28,1)


# In[68]:


# Predict the y value for each record
X_test= test_df
y_pred= model.predict(X_test)


# In[69]:


# See the last 10 predicted values
y_pred= np.argmax(y_pred, axis=1)
y_pred[:10]


# In[71]:


y_pred


# In[72]:


# The prediction on the test dataset is stored in the variable y_pred

