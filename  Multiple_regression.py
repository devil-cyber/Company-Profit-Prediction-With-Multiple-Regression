#!/usr/bin/env python
# coding: utf-8

# In[36]:


### Author Manikant Kumar ##
# Import Important Lib.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[37]:


# Datset Reading

dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values


# In[38]:


#Data Preprocessing

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x=LabelEncoder()
X[:,3]=labelencoder_x.fit_transform(X[:,3])
onehotencoder=OneHotEncoder(categorical_features=[3])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y=LabelEncoder()


# In[39]:


#Spliting The Data in Training set And Test set

from sklearn.model_selection import train_test_split


# In[40]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[42]:


#import Linear Regression Lib.

from sklearn.linear_model import LinearRegression


# In[43]:


regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[44]:


#Start Prediction

y_pred=regressor.predict(X_test)


# In[47]:


# The Predicted Value
y_pred


# In[48]:


#The Actual Value
y_test


# In[ ]:




