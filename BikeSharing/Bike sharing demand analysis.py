#!/usr/bin/env python
# coding: utf-8

# In[47]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#import missingno as msno
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

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


# In[48]:


train = pd.read_csv(r"C:\\Users\\cvveljanovski\\Desktop\\bike-sharing-demand\\train.csv")
test = pd.read_csv(r"C:\\Users\\cvveljanovski\\Desktop\\bike-sharing-demand\\test.csv")
df = train.copy()
test_df = test.copy()
df.head()


# In[49]:


df.columns.unique()


# A SHORT DESCRIPTION OF THE FEATURES.
# datetime - hourly date + timestamp
# 
# season - 1 = spring, 2 = summer, 3 = fall, 4 = winter
# 
# holiday - whether the day is considered a holiday
# 
# workingday - whether the day is neither a weekend nor holiday
# 
# weather -
# 
# 1: Clear, Few clouds, Partly cloudy
# 
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# 
# temp - temperature in Celsius
# 
# atemp - "feels like" temperature in Celsius
# 
# humidity - relative humidity
# 
# windspeed - wind speed
# 
# casual - number of non-registered user rentals initiated
# 
# registered - number of registered user rentals initiated
# 
# count - number of total rentals for that hour

# HERE ALL THE VARIABLES OR FEATURES ARE NUMERIC AND THE TARGET VARIABLE THAT WE HAVE TO PREDICT IS THE count VARIABLE. HENCE THIS IS A TYPICAL EXAMPLE OF A REGRESSION PROBLEM AS THE count VARIABLE IS CONTINUOUS VARIED

# In[50]:


#See the data types of the columns
df.info()


# In[51]:


#Check if the columns have empty cells
df.isnull().sum()


# Now lets start with the exploratory analysis. FIRST LETS EXPLORE THE DISTRIBUTION OF VARIOUS DISCRETE FEATURES LIKE weather , season etc... .

# In[52]:


# Lets check season first
df.season.value_counts()


# In[53]:


sns.factorplot(x='season', data=df, kind='count', size = 5, aspect=1.5)


# We can see that the usage of the bikes is almost equally distrobuted among the different seasons

# In[54]:


#What about holiday
df.holiday.value_counts()


# In[18]:


sns.factorplot(x='holiday', data=df, kind='count', size = 5)


# We see that the bikes are used more during days that are not holidays

# In[19]:


df.workingday.value_counts()


# In[20]:


sns.factorplot(x="workingday", data=df, kind='count', size=5)


# Here we see that the bikes are used way more during working days that during weekends or holidays

# In[21]:


df.weather.value_counts()


# In[26]:


sns.factorplot(x="weather", data=df, kind="count", size=5, aspect=2)


# Here we see that the bikes are mostly used during Clear, Few clouds, Partly cloudy weather and the least when it rains a lot

# NOW LETS SEE ALSO THE DISTRIBUTION OF THE CONTINOUS VARIABLES

# In[28]:


#Lets first check the statistics (count,mean,std,min,max. etc for each column)
df.describe()


# In[30]:


#Lets visualize the table above with box plots
sns.boxplot(data=df[["temp","atemp","humidity","windspeed", "casual","registered","count"]])
fig=plt.gcf()
fig.set_size_inches(10,10)


# In[35]:


#Using histograms
df.temp.unique()
fig, axes = plt.subplots(2,2)
axes[0,0].hist(x="temp", data=df, edgecolor="black", linewidth=2, color="red")
axes[0,0].set_title("Variation of temp")
axes[0,1].hist(x="atemp", data=df, edgecolor="black", linewidth=2, color="red")
axes[0,1].set_title("Variation of atemp")
axes[1,0].hist(x="windspeed", data=df, edgecolor="black", linewidth=2, color="red")
axes[1,0].set_title("Variation of wind speed")
axes[1,1].hist(x="humidity", data=df, edgecolor="black", linewidth=2, color="red")
axes[1,1].set_title("Variation of humidity")
fig.set_size_inches(10,10)


# In[40]:


#Now lets see the correlation matrix to see how the variables are correlated among them selves
cor_mat=df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)]=False
fig=plt.gcf()
fig.set_size_inches(15,15)
sns.heatmap(data=cor_mat, mask=mask, square=True, annot=True, cbar=True)


# From the graph above we can conclude:
# 1. Season is highly correlated with temp and atemp
# 2. Holiday is strongly inversly correlated with workingday(as expeted)
# 3. Working day is highly inversly correlated with casual ussage of the bikes(as expected)
# 4. Temps is highly correlated with count, which means that the hotter the weather the more bikers
# 5. Humidity is highly inversly correlated with count/casual/registered meaning that with higher humidity less bikers
# 6. Casual is strongly inversly correlated with workingday(as expeted)
# 7. Registered is highly correlated with count meaning we have more registered users of the bikes
# 8. Windspeed is highly infersly correlated with humidity meaning that with more wind less humid it is

# Lets do some feature engineering and get some new features while dropping some useless and less relevant features

# In[55]:


#Separating seasons in different columns
season=pd.get_dummies(df['season'],prefix='season')
df=pd.concat([df,season],axis=1)
df.head()
season=pd.get_dummies(test_df['season'],prefix='season')
test_df=pd.concat([test_df,season],axis=1)
test_df.head()


# In[56]:


# # # same for weather. this is bcoz this will enhance features.
weather=pd.get_dummies(df['weather'],prefix='weather')
df=pd.concat([df,weather],axis=1)
df.head()
weather=pd.get_dummies(test_df['weather'],prefix='weather')
test_df=pd.concat([test_df,weather],axis=1)
test_df.head()


# In[57]:


#Drop season and weather columns
df.drop(["season","weather"],inplace=True,axis=1)
df.head()
test_df.drop(['season','weather'],inplace=True,axis=1)
test_df.head()


# In[58]:


df.head()


# now most importantly split the date and time as the time of day is expected to effect the no of bikes. for eg at office hours like early mornning or evening one would expect a greater demand of rental bikes.

# In[59]:


df["hour"]=[t.hour for t in pd.DatetimeIndex(df.datetime)]
df["day"]=[t.dayofweek for t in pd.DatetimeIndex(df.datetime)]
df["month"]=[t.month for t in pd.DatetimeIndex(df.datetime)]
df["year"]=[t.year for t in pd.DatetimeIndex(df.datetime)]
df["year"]=df["year"].map({2011:0,2012:1})
df.head()


# In[60]:


test_df["hour"] = [t.hour for t in pd.DatetimeIndex(test_df.datetime)]
test_df["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_df.datetime)]
test_df["month"] = [t.month for t in pd.DatetimeIndex(test_df.datetime)]
test_df['year'] = [t.year for t in pd.DatetimeIndex(test_df.datetime)]
test_df['year'] = test_df['year'].map({2011:0, 2012:1})
test_df.head()


# In[61]:


#Drop column datetime
df.drop("datetime", axis=1,inplace=True)
df.head()


# In[62]:


#Lets check the new correlation matrix with the new features
cor_mat= df[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig=plt.gcf()
fig.set_size_inches(30,12)
sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)


# In[63]:


#Drop casual and registered since we need just the count which is the sum of them both
df.drop(['casual','registered'],axis=1,inplace=True)
df.head()


# In[64]:


#Lets see the count with hour
sns.factorplot(x="hour",y="count",data=df,kind='bar',size=5,aspect=1.5)


# We can see that the highest demand of the bikes is at 8am and between 16 and 19 in the afternoon

# In[65]:


sns.factorplot(x="month", y="count", data=df, kind="bar", size=5, aspect=1.5)


# We can see that the colder months have less bike rents than the warmer months. This is normal since the months with more rain and snow as well as with higher humidity result with less customers

# In[66]:


sns.factorplot(x="year", y="count", data=df, kind="bar", size=5)


# We can see that the second year there has been increased interest in using the bike sharing service

# In[67]:


sns.factorplot(x="day", y="count", data=df, kind="bar", size=5, aspect=1.5)


# No big difference between the days from monday to saturday, but sunday we have less rentals of bikes

# In[70]:


#Now for temp
plt.scatter(x="temp", y="count", data=df, color="red")


# This is quite difficult to unredstand so it will be better to be shown in barchart and we need to create bins for that in order to have discrete values

# In[71]:


new_df=df.copy()
new_df.temp.describe()
new_df['temp_bin']=np.floor(new_df['temp'])//5 # we use 5 to difide the vaule by 5 and select the integer part of the division
new_df['temp_bin'].unique()
# now we can visualize as follows
sns.factorplot(x="temp_bin",y="count",data=new_df,kind='bar')


# now the demand is highest for bins 6 and 7 which is about tempearure 30-35(bin 6) and 35-40 (bin 7).

# and similarly we can do for other continous variables and see how it effect the target variable.

# In[72]:


#Modeling part now


# In[73]:


df.head()


# In[75]:


#Group the columns by data type and show them as series
df.columns.to_series().groupby(df.dtypes).groups


# In[78]:


#Split the dataset into training and test
x_train, x_test, y_train, y_test = train_test_split(df.drop("count",axis=1),df['count'],test_size=0.25, random_state=42)


# In[96]:


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


# We can see that the smallest error is from the Random Forest Regressor and Bagging Regressor algorithms

# In[97]:


#Lets show the results in a table for better view
rmsle_frame=pd.DataFrame(d)
rmsle_frame


# In[98]:


#Lets show it with a graph now
sns.factorplot(y='Modelling Algo',x='RMSLE',data=rmsle_frame,kind='bar',size=5,aspect=2)


# In[99]:


sns.factorplot(x='Modelling Algo',y='RMSLE',data=rmsle_frame,kind='point',size=5,aspect=2)


# In[ ]:




