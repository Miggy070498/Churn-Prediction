#!/usr/bin/env python
# coding: utf-8

# ### Churn Prediction

# In[1]:


#Churn occurs when a customer unsubscribes 
#from a certain product or service

#Customer retention is cheaper than customer acquisition


# In[3]:


import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[7]:


df = pd.read_csv(r"C:\Users\User\Desktop\WA_Fn-UseC_-Telco-Customer-Churn.csv")


# In[8]:


df.head()


# In[9]:


df.shape


# In[10]:


df.columns.values


# In[11]:


df.isna().sum()


# In[12]:


df.describe()


# In[13]:


df["Churn"].value_counts()


# In[14]:


sns.countplot(df["Churn"])


# In[15]:


num_retained = df[df.Churn == "No"].shape[0]
num_churned = df[df.Churn == "Yes"].shape[0]

print(num_retained/(num_retained + num_churned))


# In[16]:


sns.countplot(x = "gender", hue = "Churn", data = df)
#gender has no correlation with churn count


# In[17]:


sns.countplot(x = "InternetService", hue = "Churn", data = df)
#Fiber Optic Internernet Service has highest churn count


# In[19]:


num_features = ["tenure", "MonthlyCharges"]
fig, ax = plt.subplots(1,2, figsize=(28,8))
df[df.Churn== "No"][num_features].hist(bins=20, color="blue", alpha=0.5, ax=ax)
df[df.Churn== "Yes"][num_features].hist(bins=20, color="orange", alpha=0.5, ax=ax)


# In[20]:


clean_df = df.drop("customerID", axis = 1)


# In[21]:


clean_df.shape


# In[25]:


for column in clean_df.columns:
    if clean_df[column].dtype == np.number:
        continue
    clean_df[column] = LabelEncoder().fit_transform(clean_df[column])


# In[26]:


clean_df.dtypes


# In[27]:


X = clean_df.drop("Churn", axis = 1)
y = clean_df["Churn"]

X = StandardScaler().fit_transform(X)


# In[28]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 30) 


# In[30]:


model = LogisticRegression()
model.fit(x_train, y_train)


# In[31]:


pred = model.predict(x_test)
print(pred)


# In[33]:


print(classification_report(y_test, pred))


# In[ ]:




