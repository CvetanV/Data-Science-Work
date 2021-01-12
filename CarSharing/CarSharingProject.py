#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np
from sklearn import svm
import seaborn as sns
import pandas as pd
from PIL import Image
import math
plt.style.use('seaborn')

#import the necessary modelling algos.

#classifiaction.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

#regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

#model selection
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#evaluation metrics
from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  # for classification


# In[3]:


# Load the CSV file
data = pd.read_csv("C:\\Users\\cvveljanovski\\Desktop\\carshare.csv")


# In[5]:


##Print top 5 rows
data.head()

##Check for # of empty cells
print(data.isnull().sum())

##Check mean, std etc in data
print(data.describe())

##Check the data format of the columns 
print(data.info())


# In[6]:


#drop unnecessary columns
data.drop(["Hash","event_hash","event_type","city","vendor", "minute"],inplace=True,axis=1)
data.head()


# In[7]:


## EXPLORATORY DATA ANALYSIS FOR THE VARILUS COLUMNS
# for duration, soc_delta and plate the distribution of theese continous variables 
sns.boxplot(data=data[["duration","soc_delta","plate"]])
fig=plt.gcf()
fig.set_size_inches(10,10)


# In[8]:


#Using histogramsfor the same above
data.duration.unique()
fig, axes = plt.subplots(2,2)
axes[0,0].hist(x="duration", data=data, edgecolor="black", linewidth=2, color="red")
axes[0,0].set_title("Variation of duration")
axes[0,1].hist(x="soc_delta", data=data, edgecolor="black", linewidth=2, color="red")
axes[0,1].set_title("Variation of soc_delta")
axes[1,0].hist(x="plate", data=data, edgecolor="black", linewidth=2, color="red")
axes[1,0].set_title("Variation of car plates")
fig.set_size_inches(10,10)


# In[9]:


sns.factorplot(x='month', data=data, kind='count', size = 5, aspect=1.5)


# In[ ]:


#the data we have is just for one month


# In[10]:


sns.factorplot(x='weekday', data=data, kind='count', size = 5, aspect=1.5)


# In[11]:


sns.factorplot(x='day', data=data, kind='count', size = 5, aspect=1.5)


# In[12]:


sns.factorplot(x='daytype', data=data, kind='count', size = 5, aspect=1.5)


# In[13]:


sns.factorplot(x='hour', data=data, kind='count', size = 5, aspect=1.5)


# In[10]:


# Calculate distance between start and end of the trip
import math

def distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1))         * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c

    return d


# In[11]:


dis = []
for i in range(0, 236503):
    lat1 = data["start_latitude"].iloc[i] 
    lat2 = data["end_latitude"].iloc[i]
    lon1 = data["start_longitude"].iloc[i] 
    lon2 = data["end_longitude"].iloc[i]
    dis.append(distance((lat1, lon1), (lat2, lon2)))


# In[12]:


data['distance'] = dis


# In[13]:


# What is the average distance of the trips
distance_mean = np.mean(data["distance"])
print("The mean distance of the trips in KM is:", distance_mean)
#The mean distance of the trips in KM is: 2.023611357434557

# What is the longest distance trip
distance_max = np.max(data["distance"])
print("The maximum distance of the trips is", distance_max)
#The maximum distance of the trips is 39.87549610604514

# What is the shortest distance trip
distance_min = np.min(data["distance"])
print("The minimum distance of the trips is", distance_min)
#The minimum distance of the trips is 0.0


# In[15]:


sns.boxplot(data=data[["distance"]])
fig=plt.gcf()
fig.set_size_inches(10,10)


# In[18]:


#Lets print out the map of the departures and of the arrivals of the cars
#BBox = (data.start_longitude.min(), data.start_longitude.max(), data.start_latitude.min(), data.start_latitude.max())
BBox = (9.1224, 9.2683, 45.4341, 45.5025)
BBox


# In[19]:


#Load the MAP immage
ruh_m = plt.imread('C:\\Users\\cvveljanovski\\Desktop\\mapp.png')


# In[20]:


fix, ax = plt.subplots(figsize = (20,20))
ax.scatter(data.start_longitude, data.start_latitude, zorder =1, alpha = 0.2, c = 'b', s = 0.5)
ax.set_title("Map of departures")
ax.set_xlim(BBox[0], BBox[1])
ax.set_ylim(BBox[2], BBox[3])

ax.imshow(ruh_m, zorder=0, extent = BBox, aspect = "equal")


# In[21]:


fix, ax = plt.subplots(figsize = (20,20))
ax.scatter(data.end_longitude, data.end_latitude, zorder =1, alpha = 0.2, c = "black", s = 0.5)
ax.set_title("Map of arrivals")
ax.set_xlim(BBox[0], BBox[1])
ax.set_ylim(BBox[2], BBox[3])

ax.imshow(ruh_m, zorder=0, extent = BBox, aspect = "equal")


# In[22]:


# What is the average duration of the trips
duration_mean = np.mean(data["duration"])
print("The mean duration of the trips in minutes is:", duration_mean)
#Result: 28.614607214257198 minutes
# What is the longest duration trip
duration_max = np.max(data["duration"])
print("The maximum duration of the trips is", duration_max, "in minutes or", duration_max/60, "in hours.")
# The maximum duration of the trips is 1439.866667 in minutes or 23.997777783333333 in hours.
# What is the shortest duration trip
duration_min = np.min(data["duration"])
print("The minimum duration of the trips is", duration_min, "minutes or", duration_min*60, "in seconds")
#The minimum duration of the trips is 0.483333333 minutes or 28.99999998 in seconds


# In[23]:


num_delta_0 = data["soc_delta"].value_counts(normalize=True)
num_delta_0*100
# Percentage of customers that return the car with the same amout of gas is 70.3%.


# In[24]:


#Now lets see the correlation matrix to see how the variables are correlated among them selves
cor_mat=data[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)]=False
fig=plt.gcf()
fig.set_size_inches(15,15)
sns.heatmap(data=cor_mat, mask=mask, square=True, annot=True, cbar=True)


# We can see no significant correlation the different variables

# In[17]:


df = data.copy()
#Split start_time and end_time columns in two separate columns by date and time
df['Start_Date'] = pd.to_datetime(df['start_time']).dt.date
df['Start_Time'] = pd.to_datetime(df['start_time']).dt.time
df.head()


# In[18]:


#Drop columns not usefull
df.drop(["plate","start_time","end_time","start_latitude","start_longitude","end_latitude","end_longitude","start_soc","end_soc","month","duration","daytype","soc_delta","distance","Start_Date","Start_Time"],inplace=True,axis=1)


# In[19]:


df.head()


# In[29]:


counted_df = df.groupby(["day", "hour","weekday"]).size().reset_index(name="count")


# In[30]:


counted_df


# In[31]:


counted_df.describe()


# In[79]:


#Show the count of each 
sns.factorplot(x='count', data=counted_df, kind='count', size = 5, aspect=1.5)


# In[86]:


#Split the dataset into training and test
x_train, x_test, y_train, y_test = train_test_split(counted_df.drop("count",axis=1),counted_df['count'],test_size=0.25, random_state=0)


# In[91]:


#In Models select which algorithms you want to use to train different models and using mean_squared error see which one has the smallest error
models=[KNeighborsClassifier(), LogisticRegression(), LinearSVC(), DecisionTreeClassifier(), GaussianNB(), RandomForestRegressor(),AdaBoostRegressor(),BaggingRegressor(),SVR(),KNeighborsRegressor()]
model_names=["KNeighborsClassifier","LogisticRegression", "LinearSVC", "DecisionTreeClassifier","GaussianNB", 'RandomForestRegressor','AdaBoostRegressor','BaggingRegressor','SVR','KNeighborsRegressor']
rmsle=[]
d={}
for model in range (len(models)):
    clf=models[model]
    clf.fit(x_train,y_train)
    test_pred=clf.predict(x_test)
    rmsle.append(np.sqrt(mean_squared_log_error(test_pred,y_test)))
d={'Modelling Algo':model_names,'RMSLE':rmsle}   
d


# In[92]:


#Lets show the results in a table for better view
rmsle_frame=pd.DataFrame(d)
rmsle_frame


# In[93]:


#Lets show it with a graph now
sns.factorplot(y='Modelling Algo',x='RMSLE',data=rmsle_frame,kind='bar',size=5,aspect=2)


# In[ ]:




