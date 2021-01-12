# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 21:34:58 2020

@author: cvveljanovski
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

data = pd.read_csv("dataset.csv")

#print(data.head())
#print(data.describe())
#print(data.info())

#plt.rcParams['figure.figsize'] = (20,10)
#fig,axes = plt.subplots(nrows = 2, ncols = 3)

num_features = ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE", "default_payment"]
#xaxes = num_features
#yaxes = ["Counts","Counts","Counts","Counts","Counts","Counts"]
"""
axes = axes.ravel()
for idx, ax in enumerate(axes):
    ax.hist(data[num_features[idx]].dropna(), bins=30)
    ax.set_xlabel(xaxes[idx], fontsize=15)
    ax.set_ylabel(yaxes[idx], fontsize=15)
    ax.tick_params(axis='both', labelsize=15)
"""

#print(data.columns)

output = "default_payment"

#Exploratory data analysis of the dataset
cols = [ c for c in data.columns if data.dtypes[c] != "object"]
cols.remove("ID")
cols.remove(output)
"""
c = pd.melt(data, id_vars = output, value_vars = cols)
g = sns.FacetGrid(c, hue = output, col = "variable", col_wrap = 5, sharex = False, sharey = False)
g = g.map(sns.distplot, "value", kde = True).add_legend()
"""
#How to understand the correlation among the different variables we will use the Chi squared test of independence 
#Let's make first the function that later will be used to veify the variables
def ChiSquaredTestOfIndependence( df, inputVar, Outcome_Category ):
    # Useful to have this wrapped in a function
    # The ChiSquaredTest of Independence - 
    # has a null hypothesis: the OutcomeCategory is independent of the inputVar
    # So we create a test-statistic which is a measure of the difference between 
    # "expected" i.e. what we WOULD observe if the OutcomeCategory WAS independent of the inputVar
    # "observed" i.e. what the data actually shows
    # the p-value returned is the probability of seeing this test-statistic if the null-hypothesis is true
    Outcome_Category_Table = df.groupby(Outcome_Category)[Outcome_Category].count().values
    Outcome_Category_Ratios = Outcome_Category_Table / sum(Outcome_Category_Table)
    possibleVals = df[inputVar].unique()
    observed = []
    expected = []
    for possible in possibleVals:
        countsInCategories = df[ df[ inputVar ] == possible ].groupby( Outcome_Category )[Outcome_Category].count().values
        if( len(countsInCategories) != len( Outcome_Category_Ratios ) ):
            print("Error! The class " + str( possible) +" of \'" + inputVar + "\' does not contain all values of \'" + Outcome_Category + "\'" )
            return
        elif( min(countsInCategories) < 5 ):
            print("Chi Squared Test needs at least 5 observations in each cell!")
            print( inputVar + "=" + str(possible) + " has insufficient data")
            print( countsInCategories )
            return
        else:
            observed.append( countsInCategories )   
            expected.append( Outcome_Category_Ratios * len( df[df[ inputVar ] == possible ]))
    observed = np.array( observed )
    expected = np.array( expected )
    chi_squared_stat = ((observed - expected)**2 / expected).sum().sum()
    degOfF = (observed.shape[0] - 1 ) *(observed.shape[1] - 1 ) 
    #crit = stats.chi2.ppf(q = 0.95,df = degOfF) 
    p_value = 1 - stats.chi2.cdf(x=chi_squared_stat, df=degOfF)
    print("Calculated test-statistic is %.2f" % chi_squared_stat )
    print("If " + Outcome_Category + " is independent of " + inputVar + ", this has prob %.2e of occurring" % p_value )
    #t_stat, p_val, doF, expArray = stats.chi2_contingency(observed= observed, correction=False)
    #print("Using built-in stats test: outputs")
    #print("test-statistic=%.2f, p-value=%.2f, degsOfFreedom=%d" % ( t_stat, p_val, doF ) )
    
#ChiSquaredTestOfIndependence( data, "SEX", output )
#From the result we see that there the default has a very low probability of independence(Meaning it is very dependent)

#ChiSquaredTestOfIndependence( data, "EDUCATION", output )
#There is not enough data values in education = 0 lets check how many values
#are in the rest if the classes
#print("We have %d with EDUCATION=0" % len(data.loc[ data["EDUCATION"]==0]))
#print("We have %d with EDUCATION=4" % len(data.loc[ data["EDUCATION"]==4]))
#print("We have %d with EDUCATION=5" % len(data.loc[ data["EDUCATION"]==5]))
#print("We have %d with EDUCATION=6" % len(data.loc[ data["EDUCATION"]==6]))

# Since we have 30k samples, let's just put these non-typical Education instances all into the EDUCATION=4 class and continue 
#data["EDUCATION_Corr"] = data["EDUCATION"].apply( lambda x: x if ((x>0) and (x<4)) else 4 )

#ChiSquaredTestOfIndependence(data, "EDUCATION_Corr", output)
#cols.remove("EDUCATION")
#cols.append("EDUCATION_Corr") #Remove the columnd EDUCATION and add EDUCATION_Corr in order to have values from 1 to 4 for education

#ChiSquaredTestOfIndependence(data, "MARRIAGE", output)
#From the result we see that there the default has a very low probability of independence(Meaning it is very dependent also to marriage and education)

#Quantitative vars are
quant = ["LIMIT_BAL", "AGE"]

#Qualitative but Encoded variables
qual_Enc = cols
#qual_Enc.remove("LIMIT_BAL")
#qual_Enc.remove("AGE")

#And the PAY_ variables? We can see those are important, but we'll transform the BILL_AMT and PAY_AMT variables from NT Dollars to Log(NT Dollars)

logged = []
for ii in range(1,7):
    qual_Enc.remove("PAY_AMT" + str( ii ))
    data[ "log_PAY_AMT" + str( ii )]  = data["PAY_AMT"  + str( ii )].apply( lambda x: np.log1p(x) if (x>0) else 0 )
    logged.append("log_PAY_AMT" + str( ii ) )

for ii in range(1,7):
    qual_Enc.remove("BILL_AMT" + str(ii))
    data["log_BILL_AMT" + str(ii)] = data["BILL_AMT" + str(ii)].apply(lambda x: np.log1p(x) if (x>0) else 0)
    logged.append("log_BILL_AMT" + str(ii))
""" 
f = pd.melt(data, id_vars=output, value_vars = logged)
g = sns.FacetGrid(f, hue = output, col = "variable", col_wrap = 3, sharex = False, sharey = False)
g = g.map(sns.distplot, "value", kde = True).add_legend()
"""
#It looks like higher Log PAY_AMT is associated with slightly less default.
#So now we have quant variables, qual_Enc variables and logged variables. Let's check correlations with the output variable:
"""
features = quant + qual_Enc + logged + [output]
corr = data[features].corr()
plt.subplots(figsize=(30,10))
sns.heatmap(corr, square=True, annot=True, fmt=".1f")
"""
#Lets do some predictions whether a customer will default so we separate the dataset in train and test set and we try different classfiers and compare them in performance
features = quant + qual_Enc + logged
X = data[features].values
y = data[output].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
X_train = scX.fit_transform(X_train)
X_test = scX.fit_transform(X_test)

#To evaluate the models we will use some metrics like confusion matrix and cross_validation score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
#since it is possible that the data is not linearlt separable we will try first random forest classifier and kernel-SVM
"""
#--------------------------------------------
#Random forest
#--------------------------------------------
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Accuracy on test set for Random forest = %.2f" %((cm[0,0] + cm[1,1])/len(X_test)))
scoresRF = cross_val_score(classifier, X_train, y_train, cv = 10)
print("Mean RandomForest CrossVal Accuracy on Train Set %.2f, with std=%.2f" % (scoresRF.mean(), scoresRF.std() ))
"""

#-------------- 
# kernel SVM 
#--------------
from sklearn.svm import SVC
classifier1 = SVC(kernel = "rbf")
classifier1.fit(X_train, y_train)
y_pred = classifier1.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print("Accuracy on test set for kernel SVM = %.2f" %((cm[0,0] + cm[1,1])/len(X_test)))
scoresSVC = cross_val_score(classifier1, X_train, y_train, cv = 10)
print("Mean kernel SVM CrossVal Accuracy on Train Set %.2f, with std=%.2f" % (scoresSVC.mean(), scoresSVC.std() ))

#We'll check some of the other classifiers - but we don't expect they will do better

"""
#--------------
# Logistic Regression 
#--------------
from sklearn.linear_model import LogisticRegression
classifier2 = LogisticRegression()
classifier2.fit( X_train, y_train )
y_pred = classifier2.predict( X_test )

cm = confusion_matrix( y_test, y_pred )
print("Accuracy on Test Set for LogReg = %.2f" % ((cm[0,0] + cm[1,1] )/len(X_test)))
scoresLR = cross_val_score( classifier2, X_train, y_train, cv=10)
print("Mean LogReg CrossVal Accuracy on Train Set %.2f, with std=%.2f" % (scoresLR.mean(), scoresLR.std() ))

#-------------- 
# Naive Bayes 
#--------------
from sklearn.naive_bayes import GaussianNB
classifier3 = GaussianNB()
classifier3.fit( X_train, y_train )
y_pred = classifier3.predict( X_test )
cm = confusion_matrix( y_test, y_pred )
print("Accuracy on Test Set for NBClassifier = %.2f" % ((cm[0,0] + cm[1,1] )/len(X_test)))
scoresNB = cross_val_score( classifier3, X_train, y_train, cv=10)
print("Mean NaiveBayes CrossVal Accuracy on Train Set %.2f, with std=%.2f" % (scoresNB.mean(), scoresNB.std() ))

#-------------- 
# K-NEIGHBOURS 
#--------------
from sklearn.neighbors import KNeighborsClassifier
classifier4 = KNeighborsClassifier(n_neighbors=5)
classifier4.fit( X_train, y_train )
y_pred = classifier4.predict( X_test )
cm = confusion_matrix( y_test, y_pred )
print("Accuracy on Test Set for KNeighborsClassifier = %.2f" % ((cm[0,0] + cm[1,1] )/len(X_test)))
scoresKN = cross_val_score( classifier3, X_train, y_train, cv=10)
print("Mean KN CrossVal Accuracy on Train Set Set %.2f, with std=%.2f" % (scoresKN.mean(), scoresKN.std() ))
"""

































