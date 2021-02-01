#!/usr/bin/env python
# coding: utf-8

# In[164]:


#Default Frameworks
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Preprocessing frameworks
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#Model Selection frameworks
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_validate

#ANN frameworks
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU, PReLU, ELU

#Measurements
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


# # Load Dataset

# In[53]:


data = pd.read_csv(r"C:\Users\cvveljanovski\Desktop\Learning\DataScience\Datasets\Hotel booking demand\hotel_bookings.csv")


# In[54]:


data.head()


# # Dataset details

# In[55]:


data.info()


# In[56]:


data.describe()


# In[64]:


#Check if there are missing values
data.isnull().sum()


# # Dataset cleanup and preparation

# ### 1. Empty values

# #### a. Remove columns with many missing values because it is useless to populate them with our values and then impact on the result

# In[69]:


#We are going to drop the features Agewnt and Company
data.drop(["agent"],inplace=True,axis=1)
data.drop(["company"],inplace=True,axis=1)


# #### b. Features with small number of missing values can be filled with mean value or mode value to avoid impacting the balance of the dataset

# In[59]:


# We are going to populate the missing values of the features: country and children but with what?
# Children can be populated with either mean or mode since there just 4 missing values, lets do that
data['children'] = data['children'].fillna(data['children'].mean())


# In[63]:


# What about country, lets see the different values that are present in the feature
data.country.value_counts()


# In[62]:


# Seing this we can say that if we populate the missing values with the MODE should be OK since the diffrence between the top two countries is quite big
data['country'] = data['country'].replace({np.nan:'PRT'})


# #### c. Drop duplicate features to reduce the dimensionality of the dataset

# In[65]:


# Here we don't have duplicate features


# ### 2. String values to numerical

# In[66]:


# Here all features that contain textual values will be encoded to contain numerical values since algorithms normally expect numerical input


# In[78]:


# Lets take just the features that are of type object:
str_cat = data.select_dtypes(include=["object"])


# In[79]:


# I am going to use label encoding that will substitute each string value with a specific number
str_cat = str_cat.apply(LabelEncoder().fit_transform)


# In[80]:


str_cat.head()


# In[82]:


# We should now take also the features that are numerical and concatenate with the categorical features that we have encoded
num = data.select_dtypes(exclude=["object"])
num.head()


# In[86]:


# Concatenate str_cat and num into one dataset that contains only numerical values for both categorical and continuous features
df = pd.concat([str_cat, num],axis=1)
df.head()


# ### 3. Distribution analysis of the features

# In[ ]:


# Now it will be good also to perform normalzation (i.e. to transform the values in the range from 0 to 1)
# But before we do the transformations lets analyse the distributions of the features


# In[121]:


fig, axes = plt.subplots(7,4)
axes[0,0].hist(x="arrival_date_month", data=df, edgecolor="black", linewidth=2, color="red")
axes[0,0].set_title("arrival_date_month")
axes[0,1].hist(x="hotel", data=df, edgecolor="black", linewidth=2, color="blue")
axes[0,1].set_title("hotel")
axes[0,2].hist(x="meal", data=df, edgecolor="black", linewidth=2, color="green")
axes[0,2].set_title("meal")
axes[0,3].hist(x="country", data=df, edgecolor="black", linewidth=2, color="yellow")
axes[0,3].set_title("country")

axes[1,0].hist(x="market_segment", data=df, edgecolor="black", linewidth=2, color="red")
axes[1,0].set_title("market_segment")
axes[1,1].hist(x="distribution_channel", data=df, edgecolor="black", linewidth=2, color="blue")
axes[1,1].set_title("distribution_channel")
axes[1,2].hist(x="reserved_room_type", data=df, edgecolor="black", linewidth=2, color="green")
axes[1,2].set_title("reserved_room_type")
axes[1,3].hist(x="assigned_room_type", data=df, edgecolor="black", linewidth=2, color="yellow")
axes[1,3].set_title("assigned_room_type")

axes[2,0].hist(x="deposit_type", data=df, edgecolor="black", linewidth=2, color="red")
axes[2,0].set_title("deposit_type")
axes[2,1].hist(x="customer_type", data=df, edgecolor="black", linewidth=2, color="blue")
axes[2,1].set_title("customer_type")
axes[2,2].hist(x="reservation_status", data=df, edgecolor="black", linewidth=2, color="green")
axes[2,2].set_title("reservation_status")
axes[2,3].hist(x="reservation_status_date", data=df, edgecolor="black", linewidth=2, color="yellow")
axes[2,3].set_title("reservation_status_date")

axes[3,0].hist(x="lead_time", data=df, edgecolor="black", linewidth=2, color="red")
axes[3,0].set_title("lead_time")
axes[3,1].hist(x="arrival_date_year", data=df, edgecolor="black", linewidth=2, color="blue")
axes[3,1].set_title("arrival_date_year")
axes[3,2].hist(x="arrival_date_week_number", data=df, edgecolor="black", linewidth=2, color="green")
axes[3,2].set_title("arrival_date_week_number")
axes[3,3].hist(x="arrival_date_day_of_month", data=df, edgecolor="black", linewidth=2, color="yellow")
axes[3,3].set_title("arrival_date_day_of_month")

axes[4,0].hist(x="stays_in_weekend_nights", data=df, edgecolor="black", linewidth=2, color="red")
axes[4,0].set_title("stays_in_weekend_nights")
axes[4,1].hist(x="stays_in_week_nights", data=df, edgecolor="black", linewidth=2, color="blue")
axes[4,1].set_title("stays_in_week_nights")
axes[4,2].hist(x="adults", data=df, edgecolor="black", linewidth=2, color="green")
axes[4,2].set_title("adults")
axes[4,3].hist(x="children", data=df, edgecolor="black", linewidth=2, color="yellow")
axes[4,3].set_title("children")

axes[5,0].hist(x="babies", data=df, edgecolor="black", linewidth=2, color="red")
axes[5,0].set_title("babies")
axes[5,1].hist(x="is_repeated_guest", data=df, edgecolor="black", linewidth=2, color="blue")
axes[5,1].set_title("is_repeated_guest")
axes[5,2].hist(x="previous_cancellations", data=df, edgecolor="black", linewidth=2, color="green")
axes[5,2].set_title("previous_cancellations")
axes[5,3].hist(x="previous_bookings_not_canceled", data=df, edgecolor="black", linewidth=2, color="yellow")
axes[5,3].set_title("previous_bookings_not_canceled")

axes[6,0].hist(x="booking_changes", data=df, edgecolor="black", linewidth=2, color="red")
axes[6,0].set_title("booking_changes")
axes[6,1].hist(x="days_in_waiting_list", data=df, edgecolor="black", linewidth=2, color="blue")
axes[6,1].set_title("days_in_waiting_list")
axes[6,2].hist(x="total_of_special_requests", data=df, edgecolor="black", linewidth=2, color="green")
axes[6,2].set_title("total_of_special_requests")
axes[6,3].hist(x="required_car_parking_spaces", data=df, edgecolor="black", linewidth=2, color="yellow")
axes[6,3].set_title("required_car_parking_spaces")

fig.set_size_inches(15,25)


# ### 4. Normalization and Standardization of numerical features

# In[124]:


# From the graphs above we can see that many of the features have scewed distribution of data so we need to normalize and 
# standardize the data


# In[148]:


# Drop the output feature
df_standardized = df.drop("is_canceled",axis=1)
# Store the names of the columns in a list colls
colls = df_standardized.columns
colls


# In[149]:


sc = StandardScaler()
df_standardized = sc.fit_transform(df_standardized)
df_standardized = pd.DataFrame(df_standardized)
df_standardized.columns = colls
df_standardized.head()


# # 5. Split the dataset in training and testing

# In[152]:


X = df_standardized
Y = df.is_canceled


# In[155]:


X_train, X_temp, y_train, y_temp = train_test_split(X, Y, test_size=0.5, random_state=0)


# In[161]:


X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=0)


# # 6. Building the base model before optimizing it

# In[165]:


# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(8, activation='relu', kernel_initializer='he_uniform', input_dim=29))
# We can add also dropout to prevent overfitting
classifier.add(Dropout(0.1))
# Adding the second hidden layer
classifier.add(Dense(8, activation='relu', kernel_initializer='he_uniform'))
# Adding dropout to prevent overfitting
classifier.add(Dropout(0.1))
# Adding the output layer (output_dim is 1 as we want only 1 output from the final layer.)
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='he_uniform'))


# In[166]:


# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[173]:


# Fitting the ANN to the Training set
model = classifier.fit(X_train, y_train, batch_size=10, epochs=20)


# In[174]:


print(model.history.keys())


# In[175]:


plt.plot(model.history['accuracy'])
#plt.plot(model.history['val_accuracy'])
plt.xlabel("epoch")
plt.ylabel("accuracy")
#plt.legend(['train','validation'], loc = 'upper left')
plt.show()


# In[176]:


# Fitting the ANN to the validation set
model_val = classifier.fit(X_val, y_val, batch_size=10, epochs=20)


# In[177]:


print(model.history.keys())


# In[178]:


plt.plot(model_val.history['accuracy'])
#plt.plot(model.history['val_accuracy'])
plt.xlabel("epoch")
plt.ylabel("accuracy")
#plt.legend(['train','validation'], loc = 'upper left')
plt.show()

