#!/usr/bin/env python
# coding: utf-8

# In[17]:


#Import basic frameworks
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Import learning models
from sklearn.tree import DecisionTreeClassifier

#Frameworks for SVC
from sklearn.preprocessing import StandardScaler

#Import model selection
from sklearn import model_selection
from sklearn.model_selection import train_test_split

# Import measurement metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


# In[15]:


#Read CSV file
data = pd.read_csv(r"C:\Users\cvveljanovski\Desktop\Learning\DataScience\Datasets\HotelBookingDemandDataSet\\hotel_bookings.csv")


# In[64]:


data.head()


# In[61]:


data.isnull().sum()


# In[60]:


for column in data[['children']]:
    mean = data[column].mean()
    data[column] = data[column].fillna(mean)


# In[25]:


#data = data.drop(["company", "agent", "country"], axis=1)


# # EDA

# In[34]:


sns.factorplot(data=data, x="hotel", kind="count", size = 5, aspect = 2)


# In[35]:


sns.factorplot(data=data, x="is_canceled", kind="count", size = 5, aspect = 2)


# In[32]:


sns.factorplot(data=data, x="arrival_date_month", kind="count", size = 5, aspect = 2)


# In[33]:


sns.factorplot(data=data, x="arrival_date_year", kind="count", size = 5, aspect = 2)


# In[36]:


sns.factorplot(data=data, x="arrival_date_week_number", kind="count", size = 5, aspect = 2)


# In[37]:


sns.factorplot(data=data, x="arrival_date_day_of_month", kind="count", size = 5, aspect = 2)


# In[39]:


sns.factorplot(data=data, x="adults", kind="count", size = 5, aspect = 2)


# In[62]:


sns.factorplot(data=data, x="children", kind="count", size = 5, aspect = 2)


# In[45]:


sns.factorplot(data=data, x="babies", kind="count", size = 5, aspect = 2)


# In[46]:


sns.factorplot(data=data, x="is_repeated_guest", kind="count", size = 5, aspect = 2)                 


# In[47]:


sns.factorplot(data=data, x="required_car_parking_spaces", kind="count", size = 5, aspect = 2)                        


# In[48]:


sns.factorplot(data=data, x="reservation_status", kind="count", size = 5, aspect = 2)                        


# #### Lets do a short summary from our EDA
# 1. More requested are city hotels than resort hotels
# 2. A majority of the reservations have not been canceled but there are many that have been canceled
# 3. Least popular months for reservations are November, December and January and the most popular are July and August
# 4. 2016 was the year with the most reservations
# 5. 31st day of the month is the least popular for arriving at the reservation
# 6. The reservations mostly are consisted by 2 adults, 0 children and 0 babies
# 7. Mostly are new customers, rearly are repeated guests
# 8. Most of the guests do not require parking spaces
# 9. The majority of the reservations have been used and performed check out, there are very few that did not show at the hotel on the reservation, and there are many reservations that have been canceled

# In[65]:


#Now lets see the correlation matrix to see how the variables are correlated among them selves
cor_mat=data[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)]=False
fig=plt.gcf()
fig.set_size_inches(25,15)
sns.heatmap(data=cor_mat, mask=mask, square=True, annot=True, cbar=True)


# In[79]:


#Lets take just the categorical features
cat_df = data.drop(["is_canceled", "lead_time", "arrival_date_year","arrival_date_week_number","arrival_date_day_of_month", "stays_in_weekend_nights", "stays_in_week_nights", "adults","children", "babies", "is_repeated_guest", "previous_cancellations", "previous_bookings_not_canceled", "booking_changes", "days_in_waiting_list", "adr", "required_car_parking_spaces", "total_of_special_requests"], axis=1)


# In[80]:


#Lets do the one hot encoding now on the categorical features
cat_df_one_hot = pd.get_dummies(cat_df)


# In[81]:


cat_df_one_hot.head()


# In[85]:


#Lets take just the numerical features
num_df = data.drop(["hotel", "arrival_date_month","meal","market_segment", "distribution_channel","reserved_room_type","assigned_room_type", "deposit_type", "customer_type","reservation_status"],axis = 1)


# In[88]:


num_df.head()


# In[91]:


#Concatenate the categorical and numerical features
df = pd.concat([cat_df_one_hot,num_df],axis=1)
df = df.drop(["reservation_status_date"], axis = 1)


# In[92]:


#Scale the values of the features 
Scaler = StandardScaler()
Scaler.fit(df)
Scaled_data = Scaler.transform(df)


# In[140]:


# Using PCA transform the features and take features that will be useful since we have many features now
from sklearn.decomposition import PCA
pca = PCA(n_components = 4) #(or any number that you want0
pca.fit(Scaled_data)
X_pca = pca.transform(Scaled_data)


# In[142]:


# Split the data in train and test
X = X_pca
Y = data.is_canceled
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3, random_state=42)


# In[143]:


#Define the Naive Bayes classifier algorithm
DT_model = DecisionTreeClassifier().fit(X_train, Y_train) 
#Predict
DT_predict = DT_model.predict(X_test)  


# In[144]:


#See the performance of the prediction model
print(classification_report(Y_test, DT_predict))    # generate evaluation report of NB model

