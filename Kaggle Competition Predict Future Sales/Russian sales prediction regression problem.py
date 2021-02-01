#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error

#regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#model selection
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# In[2]:


data = pd.read_csv(r"C:\\Users\\cvveljanovski\\Desktop\\Learning\\DataScience\\Datasets\\competition_predict_future_sales\\sales_train.csv")


# In[3]:


data.head()


# In[4]:


#Check for null values in the data
data.isnull().sum()


# In[5]:


#Drop not needed columns for prediction since they are missing in the testing dataset
data = data.drop(["date"], axis = 1)
data = data.drop(["item_price"], axis = 1)


# In[6]:


data.head()


# In[7]:


# Lets change the item count per day column into a sum per month per item per shop
sum_month_item = data.groupby(["date_block_num","shop_id","item_id"]).agg({'item_cnt_day': ['sum']}).reset_index()


# In[8]:


sum_month_item.head()


# In[9]:


df = sum_month_item.copy()


# In[10]:


df.info()


# ### EDA per feature

# In[11]:


#Lets see the distribution of the bought items per month
sns.factorplot(x='date_block_num', data=df, kind='count', size = 5, aspect=1.5)


# In[12]:


#Lets see the distribution of the bought items per shop
sns.factorplot(y='shop_id', data=df, kind='count', size = 10, aspect=1.5)


# ### Split the training dataset into training and testing dataset

# In[13]:


X_train, X_test, y_train, y_test = train_test_split(df.drop(["item_cnt_day"],axis=1),df['item_cnt_day'],test_size=0.3, random_state=0)


# ### Lets define and fit the model

# #### 1. KNN

# In[14]:


#model=[KNeighborsRegressor(), LinearRegression(), SVR(), RandomForestRegressor() ,AdaBoostRegressor(),BaggingRegressor(),RidgeCV()]
clf = AdaBoostRegressor()

#Define the model and the GridSearchCV tool
model = AdaBoostRegressor()
estimator = model
search = GridSearchCV(
    estimator = model,
    param_grid = {
  'n_estimators':(10, 30, 50),
  'learning_rate':(0.01, 0.03)
 },
    n_jobs=-1,
    scoring="r2",
    cv=10,
    verbose=3
)
Grid_search = search.fit(X_train, y_train)
Accuracy = search.best_score_
Grid_search.best_params_


# In[15]:


clf = AdaBoostRegressor(n_estimators = 10, learning_rate = 0.01)
rmse = []
clf.fit(X_train,y_train)
test_pred=clf.predict(X_test)
rmse.append(np.sqrt(mean_squared_error(test_pred,y_test)))
d={'RMSE':rmse}   
d


# In[ ]:





# In[16]:


#Import the official test dataset
test_data = pd.read_csv(r"C:\\Users\\cvveljanovski\\Desktop\\Learning\\DataScience\\Datasets\\competition_predict_future_sales\\test.csv")


# In[17]:


pred = clf.predict(test_data)


# In[18]:


pred = pd.DataFrame(pred)


# In[19]:


pred.to_csv('C:\\Users\\cvveljanovski\\Desktop\\Learning\\DataScience\\Datasets\\competition_predict_future_sales\\pred.csv')


# In[ ]:




