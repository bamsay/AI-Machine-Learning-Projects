#!/usr/bin/env python
# coding: utf-8

# # Import Modules

# In[76]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# # Loading the Dataset

# In[77]:


df = pd.read_csv("C:/Users/Bamidele David/Desktop/Darlytics Internship/Python Projects/Loan_Dataset.csv")


# In[78]:


df.head()


# In[79]:


df.describe()


# In[80]:


df.info()


# # Preprocessing the Dataset

# In[81]:


### find the null values
df.isnull().sum()


# In[82]:


### find duplicate values
df.duplicated().sum()


# # Exploratory Data Analysis

# In[83]:


### find the number of borrowers according to their loan purpose
### The highest purpose of loan is for debt_consolidation.

plt.figure(figsize=(15, 6))
sns.countplot(data=df, x='purpose')
plt.title('Count of Loan Purpose')
plt.xlabel('Purpose')
plt.ylabel('Count')
plt.show()


# In[84]:


### Show the number of credit policy holders according to criteria
### 1: customer meets the credit underwriting criteria of LendingClub.com
### 0: Customer does not meet the criteria

plt.figure(figsize=(6, 6))
sns.countplot(data=df, x='credit.policy')
plt.title('Credit underwriting criteria')
plt.xlabel('credit.policy')
plt.ylabel('Count')
plt.show()


# In[85]:


sns.displot(data = df, x = 'int.rate', hue='not.fully.paid')


# In[86]:


sns.displot(data = df, x = 'not.fully.paid')


# In[87]:


other_cols = df.drop(['purpose', 'not.fully.paid'], axis=1)
other_cols = other_cols.columns


# In[88]:


plt.figure(figsize=(13, 10))
for i, col in enumerate(other_cols):
    i += 1
    plt.subplot(4, 3, i)
    sns.histplot(data=df, x=col, hue='not.fully.paid', legend=False)
    plt.ylabel("")
    plt.tight_layout()


# # Correlation Matrix

# In[64]:


# To understand the overall relationship among variables
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),  annot=True)


# In[89]:


df.groupby(['purpose','not.fully.paid']).count()


# In[94]:


### Drop unecessary Column

df.head()


# # Train-Test Split

# In[95]:


### Specify input and output attribute

X = df.drop('not.fully.paid', axis=1)
y = df['not.fully.paid']


# In[96]:


from sklearn.model_selection import train_test_split


# In[116]:


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 0)


# # Model Training

# In[145]:


from sklearn.preprocessing import StandardScaler
ss= StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)


# In[146]:


### Classify function

def classify(model, x, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 9)
    model.fit(x_train, y_train)
    print("Accuracy is", model.score(x_test, y_test)*100)
    


# In[147]:


from sklearn.linear_model import LogisticRegression   ### This is the first model we developed with 84.74% accuracy.
model = LogisticRegression()
classify(model, X,y)


# In[148]:


from sklearn.naive_bayes import GaussianNB    ### We developed another model for comparison and got similar accuracy of 84.74%
NBClassifier = GaussianNB()
NBClassifier.fit(x_train,y_train)
classify(model, X,y)


# # Testing our Data using the Model

# In[149]:


### We created a new csv file with the details of the 10 individuals and performed preprocessing.

testdata = pd.read_csv("C:/Users/Bamidele David/Desktop/Darlytics Internship/Python Projects/Test_Data_Darlytics_Pythonproject.csv")


# In[150]:


testdata.head()


# In[126]:


testdata.info()


# In[127]:


testdata.isnull().sum()


# In[141]:


### Lets select the columns we need into a variable

test = testdata.iloc[:, np.r_[0,2:13]].values
test


# In[151]:


test= ss.fit_transform(test)


# In[153]:


### Now we test our test data using the Gaussian Model we created

pred = NBClassifier.predict(test)


# In[154]:


pred


# # Conclusion

# In[ ]:


Where 0 represents that the customer will not pay back the loan 
& 1 represents customers that will pay back the loan, Our model 
predicted that none of the 10 customers will pay back their loan.

