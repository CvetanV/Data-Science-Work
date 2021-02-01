#!/usr/bin/env python
# coding: utf-8

# In[91]:


#Standard libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as md

#Preprocessing frameworks
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

#Model selection frameworks
from sklearn import model_selection
from sklearn.model_selection import train_test_split, cross_validate

#Algorithms for the model
from sklearn.ensemble import RandomForestRegressor

#Measurements for regression problem
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score


# # Load Dataset

# In[95]:


data = pd.read_csv(r"C:\Users\cvveljanovski\Desktop\Learning\DataScience\Datasets\Covid-19_on_economy_per_nation\transformed_data.csv")


# In[96]:


data.head()


# # Dataset details

# ### Description of the dataset
# HDI - human_development_index,
# TC - total cases,
# TD - total deaths,
# STI - stringency_index,
# Pop - Population,
# GDPCAP - GDP per capita,

# In[7]:


data.describe()


# ### Lets understand the features of the dataset in order to understand the problem that we want to analyze

# #### 1. Human Development Indexs (HDI)
# The HDI was created to emphasize that people and their capabilities should be the ultimate criteria for assessing the development of a country, not economic growth alone. The HDI can also be used to question national policy choices, asking how two countries with the same level of GNI per capita can end up with different human development outcomes. These contrasts can stimulate debate about government policy priorities. The Human Development Index (HDI) is a summary measure of average achievement in key dimensions of human development: a long and healthy life, being knowledgeable and have a decent standard of living. The HDI is the geometric mean of normalized indices for each of the three dimensions.
# 
# The HDI simplifies and captures only part of what human development entails. It does not reflect on inequalities, poverty, human security, empowerment, etc. The HDRO offers the other composite indices as broader proxy on some of the key issues of human development, inequality, gender disparity and poverty.

# #### 2. Stringency Index (STI)
# It is among the metrics being used by the Oxford COVID-19 Government Response Tracker.
# The Tracker involves a team of 100 Oxford community members who have continuously updated a database of 17 indicators of government response.
# These indicators examine containment policies such as school and workplace closings, public events, public transport, stay-at-home policies.
# The Stringency Index is a number from 0 to 100 that reflects these indicators. A higher index score indicates a higher level of stringency.

# #### 3. GDP Per Capita (GDPCAP)
# A country's GDP or gross domestic product is calculated by taking into account the monetary worth of a nation's goods and services after a certain period of time, usually one year. It's a measure of economic activity.

# In[26]:


data.info()


# # Dataset cleanup and preparation

# In[13]:


#Check if there are missing values
data.isnull().sum()


# #### We will populate the missing values with the mean value of the column to avoid impacting the distribution

# In[12]:


data['HDI'] = data['HDI'].fillna(data['HDI'].mean())


# #### Drop duplicate features to reduce the dimensionality of the dataset

# In[14]:


data.drop(["COUNTRY"],inplace=True,axis=1)


# In[ ]:


# We will split the date feature in three features date, month and day
data['DATE'] = pd.to_datetime(data['DATE'], errors='coerce') # Convert the feature from type object to type date
data["day"] = data["DATE"].dt.day
data["month"] = data["DATE"].dt.month
data["year"] = data["DATE"].dt.year
data.drop(["DATE"],inplace=True,axis=1) # Now we can drop the DATE feature since it is duplicate and we don't need it


# In[53]:


# Lets label encode the feature with string values CODE

# Store the categorical feature in one dataset
str_cat = data.select_dtypes(include=["object"])

# Store the numerical features in another dataset
num = data.select_dtypes(exclude=["object"])

# I am going to use label encoding that will substitute each string value with a specific number
str_cat = str_cat.apply(LabelEncoder().fit_transform)

# Concatenate str_cat and num into one dataset that contains only numerical values for both categorical and continuous features
df = pd.concat([str_cat, num],axis=1)
df.head()


# # Data analysis

# In[43]:


# top Countries ordered by the HDI feature
hdi = data.sort_values(by='HDI', ascending=False)[:5000]
figure = plt.figure(figsize=(15,6))
sns.barplot(y=hdi.CODE, x=hdi.HDI)
plt.xticks()
plt.xlabel('HDI')
plt.ylabel('COUNTRY')
plt.title('Countplot of country by HDI')
plt.show()


# In[52]:


# Countries ordered by the STI feature
sti = data.sort_values(by='STI', ascending=False)[:]
figure = plt.figure(figsize=(15,35))
sns.barplot(y=sti.CODE, x=sti.STI)
plt.xticks()
plt.xlabel('STI')
plt.ylabel('COUNTRY')
plt.title('Countplot of country by STI')
plt.show()


# In[50]:


# Countries ordered by the Pop feature
pop = data.sort_values(by='POP', ascending=False)[:]
figure = plt.figure(figsize=(15,35))
sns.barplot(y=pop.CODE, x=pop.POP)
plt.xticks()
plt.xlabel('POP')
plt.ylabel('COUNTRY')
plt.title('Countplot of country by population')
plt.show()


# In[97]:


fig, ax = plt.subplots(figsize = (20, 6))

sns.lineplot(data=data, x='DATE', y='TC', hue='COUNTRY', legend=False, palette='Accent_r')

ax.set_title("Total Cases by countries")
ax.set_xlabel("Date")
ax.set_ylabel("Counts")

ax.xaxis.set_major_formatter(md.DateFormatter('%Y-%m-%d'))


# In[98]:


def impact(x):
    y = data[['CODE','COUNTRY','DATE','HDI','TC','TD','STI','POP','GDPCAP']][data["COUNTRY"] == x]
    y = y.sort_values(by="CODE",ascending=False)
    return y.head(15)


# In[99]:


impact("Macedonia")


# In[100]:


HDI= data[data['COUNTRY']=='Macedonia'].groupby(['DATE']).agg({'HDI':['sum']})
GDPCAP = data[data['COUNTRY']=='Macedonia'].groupby(['DATE']).agg({'GDPCAP':['sum']})
total1= HDI.join(GDPCAP)

TD= data[data['COUNTRY']=='Macedonia'].groupby(['DATE']).agg({'TD':['sum']})
TC = data[data['COUNTRY']=='Macedonia'].groupby(['DATE']).agg({'TC':['sum']})
total2= TD.join(TC)

POP= data[data['COUNTRY']=='Macedonia'].groupby(['DATE']).agg({'POP':['sum']})
STI = data[data['COUNTRY']=='Macedonia'].groupby(['DATE']).agg({'STI':['sum']})
total3= POP.join(STI)

plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
total1.plot(ax=plt.gca(), title='Macedonia')
plt.ylabel("Counts", size=13)

plt.subplot(2, 2, 2)
total2.plot(ax=plt.gca(), title='Macedonia')
plt.ylabel("Counts", size=13)


plt.subplot(2, 2, 3)
total3.plot(ax=plt.gca(), title='Macedonia')
plt.ylabel("Counts", size=13)


# In[108]:


# Unoftunately we don't have the value for STI for Macedonia, but we can see that the deaths and the cases increases over the time


# In[102]:


impact("Italy")


# In[103]:


HDI= data[data['COUNTRY']=='Italy'].groupby(['DATE']).agg({'HDI':['sum']})
GDPCAP = data[data['COUNTRY']=='Italy'].groupby(['DATE']).agg({'GDPCAP':['sum']})
total1= HDI.join(GDPCAP)

TD= data[data['COUNTRY']=='Italy'].groupby(['DATE']).agg({'TD':['sum']})
TC = data[data['COUNTRY']=='Italy'].groupby(['DATE']).agg({'TC':['sum']})
total2= TD.join(TC)

POP= data[data['COUNTRY']=='Italy'].groupby(['DATE']).agg({'POP':['sum']})
STI = data[data['COUNTRY']=='Italy'].groupby(['DATE']).agg({'STI':['sum']})
total3= POP.join(STI)

plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
total1.plot(ax=plt.gca(), title='Italy')
plt.ylabel("Counts", size=13)

plt.subplot(2, 2, 2)
total2.plot(ax=plt.gca(), title='Italy')
plt.ylabel("Counts", size=13)


plt.subplot(2, 2, 3)
total3.plot(ax=plt.gca(), title='Italy')
plt.ylabel("Counts", size=13)


# In[109]:


# Italy is slowely decreasing the number of new cases and deaths and as we can see the STI was impacted with the start of the pandemic


# In[104]:


impact("China")


# In[105]:


HDI= data[data['COUNTRY']=='China'].groupby(['DATE']).agg({'HDI':['sum']})
GDPCAP = data[data['COUNTRY']=='China'].groupby(['DATE']).agg({'GDPCAP':['sum']})
total1= HDI.join(GDPCAP)

TD= data[data['COUNTRY']=='China'].groupby(['DATE']).agg({'TD':['sum']})
TC = data[data['COUNTRY']=='China'].groupby(['DATE']).agg({'TC':['sum']})
total2= TD.join(TC)

POP= data[data['COUNTRY']=='China'].groupby(['DATE']).agg({'POP':['sum']})
STI = data[data['COUNTRY']=='China'].groupby(['DATE']).agg({'STI':['sum']})
total3= POP.join(STI)

plt.figure(figsize=(15,10))
plt.subplot(2, 2, 1)
total1.plot(ax=plt.gca(), title='China')
plt.ylabel("Counts", size=13)

plt.subplot(2, 2, 2)
total2.plot(ax=plt.gca(), title='China')
plt.ylabel("Counts", size=13)


plt.subplot(2, 2, 3)
total3.plot(ax=plt.gca(), title='China')
plt.ylabel("Counts", size=13)


# In[110]:


# China as the first coutry to be impacted by Covid-19 has stabilized the number of deaths and new cases and as we can see
# the STI index was impacted but in the last months it fell down back to its previous values


# In[27]:


#Now it will be good also to perform normalzation (i.e. to transform the values in the range from 0 to 1)
#But before we do the transformations lets analyse the distributions of the features


# In[31]:


fig, axes = plt.subplots(2,3)
axes[0,0].hist(x="HDI", data=data, edgecolor="black", linewidth=2, color="red")
axes[0,0].set_title("HDI       ")
axes[0,1].hist(x="TC", data=data, edgecolor="black", linewidth=2, color="blue")
axes[0,1].set_title("TC")
axes[0,2].hist(x="TD", data=data, edgecolor="black", linewidth=2, color="green")
axes[0,2].set_title("TD")

axes[1,0].hist(x="STI", data=data, edgecolor="black", linewidth=2, color="red")
axes[1,0].set_title("STI")
axes[1,1].hist(x="POP", data=data, edgecolor="black", linewidth=2, color="blue")
axes[1,1].set_title("POP")
axes[1,2].hist(x="GDPCAP", data=data, edgecolor="black", linewidth=2, color="green")
axes[1,2].set_title("GDPCAP")
fig.set_size_inches(15,10)


# #### As we can see the features are not normally distributed se we should normalize them

# In[57]:


# Store the names of the columns in a list colls
df_std = df.drop("GDPCAP",axis=1)
std_col = df_std.columns
std_col


# In[58]:


sc = StandardScaler()
df_std = sc.fit_transform(df_std)
df_std = pd.DataFrame(df_std)
df_std.columns = std_col
df_std.head()


# # Split the dataset in training and testing

# In[61]:


X = df_std
Y = data.GDPCAP


# In[66]:


X_train, X_temp, Y_train, Y_temp = train_test_split(X,Y, test_size = 0.5, random_state=0)


# In[67]:


X_val, X_test, Y_val, Y_test = train_test_split(X_temp,Y_temp, test_size = 0.4, random_state=0)


# # Building the base regression model for prediction before optimizing it

# In[79]:


# Define and train model then predict
clf = RandomForestRegressor(n_estimators = 100)
model=clf.fit(X_train,Y_train)
val_pred=clf.predict(X_val)

# Check the performance of the model
rmse = []
rmse.append(np.sqrt(mean_squared_error(val_pred,Y_val)))
r2 = []
r2.append(r2_score(val_pred,Y_val))
d={'RMSE':rmse}   
d1={'R2': r2}
print(d,d1)


# #### This model is pretty good since we have an R squared value close to 1 and very low RMSE value but lets try to optimize it

# In[80]:


import eli5
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(model, random_state=1).fit(X_train, Y_train)
eli5.show_weights(perm, feature_names = X_train.columns.tolist())


# In[ ]:


# Here we can see that the features that have the biggest impact of predicting the GDP per capita value are "Population" and "HDI"
# So we are going to take those two features now for our new model


# In[81]:


X_new = df_std[["POP", "HDI"]]


# In[84]:


X_n_train, X_n_temp, Y_n_train, Y_n_temp = train_test_split(X_new,Y, test_size = 0.5, random_state=0)


# In[86]:


X_n_val, X_n_test, Y_n_val, Y_n_test = train_test_split(X_n_temp, Y_n_temp, test_size = 0.5, random_state=0)


# In[87]:


# Define and train model then predict
clf_n = RandomForestRegressor(n_estimators = 100)
model_n=clf_n.fit(X_n_train,Y_n_train)
val_pred_n=clf_n.predict(X_n_val)

# Check the performance of the model
rmse = []
rmse.append(np.sqrt(mean_squared_error(val_pred_n,Y_n_val)))
r2 = []
r2.append(r2_score(val_pred_n,Y_n_val))
d={'RMSE':rmse}   
d1={'R2': r2}
print(d,d1)


# In[ ]:


# Here we can see that the R2 value is pretty much 1 and the Root mean squared error is almost 0, this means that we
# optimized our model bu selecting the best features for the prediction

