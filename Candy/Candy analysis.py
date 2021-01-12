#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Import the frameworks
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#Import the algorithms
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


#Load the CSV file in a DF
data = pd.read_csv(r"C:\\Users\\cvveljanovski\\Desktop\\Learning\\DataScience\\Candy\\candy-data.csv")


# In[4]:


data


# In[46]:


# Order the candies by win percentage
Top10=data.sort_values(by=['winpercent'], ascending=False).head(10)


# In[47]:


#Print the top 10 candies
Top10


# In[48]:


Bottom10=data.sort_values(by=['winpercent'], ascending=False).tail(10)


# In[49]:


#Print the bottom 10 candies
Bottom10


# In[60]:


#Using histograms
Top10.chocolate.unique()
fig, axes = plt.subplots(3,2)
axes[0,0].hist(x="chocolate", data=Top10, edgecolor="black", linewidth=2, color="red")
axes[0,0].set_title("chocolate")
axes[0,1].hist(x="fruity", data=Top10, edgecolor="black", linewidth=2, color="red")
axes[0,1].set_title("fruity")
axes[1,0].hist(x="caramel", data=Top10, edgecolor="black", linewidth=2, color="red")
axes[1,0].set_title("caramel")
axes[1,1].hist(x="peanutyalmondy", data=Top10, edgecolor="black", linewidth=2, color="red")
axes[1,1].set_title("peanutyalmondy")
axes[2,0].hist(x="crispedricewafer", data=Top10, edgecolor="black", linewidth=2, color="red")
axes[2,0].set_title("crispedricewafer")
axes[2,1].hist(x="bar", data=Top10, edgecolor="black", linewidth=2, color="red")
axes[2,1].set_title("bar")
fig.set_size_inches(10,10)


# In[62]:


#Using histograms
Bottom10.chocolate.unique()
fig, axes = plt.subplots(3,2)
axes[0,0].hist(x="chocolate", data=Bottom10, edgecolor="black", linewidth=2, color="red")
axes[0,0].set_title("chocolate")
axes[0,1].hist(x="fruity", data=Bottom10, edgecolor="black", linewidth=2, color="red")
axes[0,1].set_title("fruity")
axes[1,0].hist(x="caramel", data=Bottom10, edgecolor="black", linewidth=2, color="red")
axes[1,0].set_title("caramel")
axes[1,1].hist(x="peanutyalmondy", data=Bottom10, edgecolor="black", linewidth=2, color="red")
axes[1,1].set_title("peanutyalmondy")
axes[2,0].hist(x="crispedricewafer", data=Bottom10, edgecolor="black", linewidth=2, color="red")
axes[2,0].set_title("crispedricewafer")
axes[2,1].hist(x="bar", data=Bottom10, edgecolor="black", linewidth=2, color="red")
axes[2,1].set_title("bar")
fig.set_size_inches(10,10)


# Which qualities are associated with higher rankings?
# From the graphs above we can see that a candy will win if it has:
# 1. Chocolate
# 2. No Fruity taste
# 3. Peanutty taste
# 4. Has a bar form

# In[61]:


data.columns.unique()


# A short description of the columns
# 1. Competitior name is the name of the company that makes the candy
# 2. Chocolate tells if there is chocolate in the candy
# 3. Fruity tells if there is a fruity taste in the candy
# 4. Caramel tells if there is caramel in the candy
# 5. peanutyalmondy tells if there is peanutyalmondy taste in the candy
# 6. nougat tells if there is a nougat in the candy
# 7. crispedricewafer if there is crispedricewafer in the candy
# 8. hard tells if the candy is hard
# 9. bar tells if the form of the candy is bar
# 10. pluribus tells if the form is pluribus
# 11. sugarpercent tells the percentage of sugar in the candy
# 12. pricepercent The unit price percentile compared to the rest of the set.
# 13. winpercent tells how many times the candy won when paired with another candy on the poll

# 1. Which qualities are associated with higher rankings?
# 2. What’s the most popular candy? Least popular?

# In[6]:


#Check the datatype of the columns
data.info()


# In[7]:


#Check if there are empty value cells
data.isnull().sum()


# Lets do the EDA(exploratory data analysis)

# In[8]:


#Lets see the distribution of the candies with and without chocolate
sns.factorplot(x='chocolate', data=data, kind='count', size = 5, aspect=1.5)


# We see there are more candies without chocolate

# In[9]:


#Lets see the distribution of the candies with and without fruity taste
sns.factorplot(x='fruity', data=data, kind='count', size = 5, aspect=1.5)


# We see there are more candies without fruit

# In[10]:


#Lets see the distribution of the candies with and without caramel
sns.factorplot(x='caramel', data=data, kind='count', size = 5, aspect=1.5)


# We see there are way more candies without caramel

# In[11]:


#Lets see the distribution of the candies with and without peanutyalmondy taste
sns.factorplot(x='peanutyalmondy', data=data, kind='count', size = 5, aspect=1.5)


# We see there are way more candies without peanutyalmondy taste

# In[12]:


#Lets see the distribution of the candies with and without nougat
sns.factorplot(x='nougat', data=data, kind='count', size = 5, aspect=1.5)


# We see there are a lot more candies without nugat

# In[13]:


#Lets see the distribution of the candies with and without crispedricewafer
sns.factorplot(x='crispedricewafer', data=data, kind='count', size = 5, aspect=1.5)


# We see there are a lot more candies without crispedricewafer

# In[14]:


#Lets see the distribution of the candies that are hard
sns.factorplot(x='hard', data=data, kind='count', size = 5, aspect=1.5)


# There are more soft than hard candies

# In[15]:


#Lets see the distribution of the candies with bar shape
sns.factorplot(x='bar', data=data, kind='count', size = 5, aspect=1.5)


# We see that more candies don't have the shape of bar

# In[16]:


#Lets see the distribution of the pluribus candies
sns.factorplot(x='pluribus', data=data, kind='count', size = 5, aspect=1.5)


# The distribution is quite equal

# In[17]:


#Lets visualize the mean, min, max etc for sugar percent and price percent of the candies
sns.boxplot(data=data[["sugarpercent","pricepercent"]])
fig=plt.gcf()
fig.set_size_inches(10,10)


# In[18]:


data.describe()


# In[19]:


#Now lets see the correlation matrix to see how the variables are correlated among them selves
cor_mat=data[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)]=False
fig=plt.gcf()
fig.set_size_inches(15,15)
sns.heatmap(data=cor_mat, mask=mask, square=True, annot=True, cbar=True)


# From the above figure we can see that:
# 1. Chocolate has strong inverse correlation with fruity
# 2. Chocolate has high correlation with bar
# 3. Chocolate has high correlation with win percentage which means it has won more times
# 4. Fruity has high inverse correlation with bar
# 5. Fruity has high inverse correlation with pricepercent which means that the fruity candy is less expensive
# 6. Peanutyalmondy has high correlation with win percent 
# 7. Nougat has high correlation with bar
# 8. Crispedricewafer has high correlation with bar
# 9. Bar has inverse correlation with pluribus
# 10. Bar has high correlation with pricepercent
# 11. Bar has high correlation with winpercent

# In[20]:


#2. What’s the most popular candy?
num = data['winpercent'].max()
data.loc[data['winpercent'] == num, 'competitorname']
#Result is "ReeseÕs Peanut Butter cup"


# In[21]:


#2. What’s the least popular candy?
out = data['winpercent'].min()
data.loc[data['winpercent'] == 22.445341, 'competitorname']
#Result is "Nik L Nip"


# In[22]:


#Drop the column with textual data
df = data.drop("competitorname", axis=1)
df.head()


# In[24]:


#Split the dataset into training and test
x_train, x_test, y_train, y_test = train_test_split(df.drop("winpercent",axis=1),df['winpercent'],test_size=0.25, random_state=0)


# Bellow we will perform a regression test with different algorithms and we will try different error measurement methods:

# In[63]:


#Using mean_squared_log_error lets see which algorithms will work the best
#This is a regression problem so we have to use regression algorithms
#In Models select which algorithms you want to use to train different models and using mean_squared error see which one has the smallest error
models=[SVR(),KNeighborsRegressor(),RidgeCV(),Lasso(),BaggingRegressor(),GradientBoostingRegressor(), KNeighborsRegressor()]
model_names=["SVR","KNeighborsRegressor","RidgeCV","Lasso","BaggingRegressor","GradientBoostingRegressor", "KNeighborsRegressor"]
rmsle=[]
d={}
for model in range (len(models)):
    clf=models[model]
    clf.fit(x_train,y_train)
    test_pred=clf.predict(x_test)
    rmsle.append(np.sqrt(mean_squared_log_error(test_pred,y_test)))
d={'Modelling Algo':model_names1,'RMSLE':rmsle}   
d


# In[64]:


#Lets show the results in a table for better view
rmsle_frame=pd.DataFrame(d)
rmsle_frame


# In[65]:


#Lets show it with a graph now
sns.factorplot(y='Modelling Algo',x='RMSLE',data=rmsle_frame,kind='bar',size=5,aspect=2)


# We can see that SVR, KNeighbors Regressor, RidgeCV and Lasso algorithms are performing the best

# In[66]:


#Using mean_squared_error lets see which algorithm will work the best
#This is a regression problem so we have to use regression algorithms
#In Models select which algorithms you want to use to train different models and using mean_squared error see which one has the smallest error
models1=[SVR(),BaggingRegressor(),KNeighborsRegressor(),LinearRegression(),Ridge(),RidgeCV(),Lasso(),RandomForestRegressor(),BaggingRegressor(),GradientBoostingRegressor(),AdaBoostRegressor(), KNeighborsRegressor()]
model_names1=["SVR","BaggingRegressor","KNeighborsRegressor","LinearRegression","Ridge","RidgeCV","Lasso","RandomForestRegressor","BaggingRegressor","GradientBoostingRegressor","AdaBoostRegressor", "KNeighborsRegressor"]
mse=[]
d1={}
for model1 in range (len(models1)):
    clf1=models1[model1]
    clf1.fit(x_train,y_train)
    test_pred1=clf1.predict(x_test)
    mse.append(np.sqrt(mean_squared_error(test_pred1,y_test)))
d1={'Modelling Algo':model_names1,'MSE':mse}   
d1


# In[67]:


#Lets show the results in a table for better view
mse_frame=pd.DataFrame(d1)
mse_frame


# In[69]:


#Lets show it with a graph now
sns.factorplot(y='Modelling Algo',x='MSE',data=mse_frame,kind='bar',size=5,aspect=2)


# We can see that Random Forest Regressor and Bagging Regressor are performing the best

# In[75]:


#Using r2_score lets see which algorithm will work the best
#This is a regression problem so we have to use regression algorithms
#In Models select which algorithms you want to use to train different models and using mean_squared error see which one has the smallest error
models2=[BaggingRegressor(),LinearRegression(),RandomForestRegressor(),BaggingRegressor(),GradientBoostingRegressor()]
model_names2=["BaggingRegressor","LinearRegression","RandomForestRegressor","BaggingRegressor","GradientBoostingRegressor"]
R2=[]
d2={}
for model2 in range (len(models2)):
    clf2=models2[model2]
    clf2.fit(x_train,y_train)
    test_pred2=clf2.predict(x_test)
    R2.append(np.sqrt(r2_score(test_pred2,y_test)))
d2={'Modelling Algo':model_names2,'R2':R2}   
d2


# In[76]:


#Lets show the results in a table for better view
R2_frame=pd.DataFrame(d2)
R2_frame


# In[77]:


#Lets show it with a graph now
sns.factorplot(y='Modelling Algo',x='R2',data=R2_frame,kind='bar',size=5,aspect=2)


# We can see that Random Forest Regressor performs the best

# In[79]:


#Using mean_absolute_error lets see which algorithm will work the best
#This is a regression problem so we have to use regression algorithms
#In Models select which algorithms you want to use to train different models and using mean_squared error see which one has the smallest error
models3=[SVR(),BaggingRegressor(),KNeighborsRegressor(),LinearRegression(),Ridge(),RidgeCV(),Lasso(),RandomForestRegressor(),BaggingRegressor(),GradientBoostingRegressor(),AdaBoostRegressor(), KNeighborsRegressor()]
model_names3=["SVR","BaggingRegressor","KNeighborsRegressor","LinearRegression","Ridge","RidgeCV","Lasso","RandomForestRegressor","BaggingRegressor","GradientBoostingRegressor","AdaBoostRegressor", "KNeighborsRegressor"]
mae=[]
d3={}
for model3 in range (len(models3)):
    clf3=models3[model3]
    clf3.fit(x_train,y_train)
    test_pred3=clf3.predict(x_test)
    mae.append(np.sqrt(mean_absolute_error(test_pred3,y_test)))
d3={'Modelling Algo':model_names3,'MAE':mae}   
d3


# In[80]:


#Lets show the results in a table for better view
mae_frame=pd.DataFrame(d3)
mae_frame


# In[81]:


#Lets show it with a graph now
sns.factorplot(y='Modelling Algo',x='MAE',data=mae_frame,kind='bar',size=5,aspect=2)


# We can see that Random Forest Regressor works the best

# In[ ]:




