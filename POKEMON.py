#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv(r'\pokemon-additional-variables.csv')
df.head(5)


# In[3]:


df.shape


# In[4]:


df.info()


# In[6]:


df.describe()


# In[9]:


df.boxplot()
plt.xticks(rotation=90)


# In[10]:


d = df.columns


# In[11]:


cor_matrix = df.corr()


# In[13]:


sb.heatmap(cor_matrix,annot=True,vmin=-1,vmax=1)


# In[14]:


df.dtypes


# In[15]:


df = df.drop_duplicates()


# In[16]:


df.get_dtype_counts()


# In[17]:


df.get_values()


# In[18]:


df.get_ftype_counts()


# In[20]:


df.ftypes


# In[22]:


df.base_happiness.hist()


# In[23]:


df.is_legendary.value_counts()


# In[24]:


df.items()


# In[25]:


df.sp_attack.hist()


# In[26]:


df.sp_defense.hist()


# In[27]:


df.type2.mode()


# In[30]:


df.weight_kg.median()


# In[35]:


df.percentage_male.median()
df.height_m.median()


# In[36]:


df.type2.fillna(df.type2.mode()[0],inplace=True)
df.weight_kg.fillna(df.weight_kg.median(),inplace=True)
df.percentage_male.fillna(df.percentage_male.median(),inplace=True)
df.height_m.fillna(df.height_m.median(),inplace=True)


# In[37]:


df.isnull().sum()


# In[38]:


sb.countplot(x="against_psychic",data=df)


# In[39]:


sb.factorplot(x = "against_psychic",y="against_water",data=df)


# In[42]:


sb.jointplot(x = "against_water",y="against_bug",data=df)


# In[45]:


sb.lmplot(x="base_total",y="attack",data=df)


# In[48]:


plt.scatter(x="base_total",y='against_bug',data=df)


# In[63]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[85]:


df["abilities"]=le.fit_transform(df["abilities"])
df["classfication"]=le.fit_transform(df["classfication"])
df["japanese_name"]=le.fit_transform(df["japanese_name"])
df["name"]=le.fit_transform(df["name"])
df["type1"]=le.fit_transform(df["type1"])
df["type2"]=le.fit_transform(df["type2"])
df["capture_rate"]=le.fit_transform(df["capture_rate"])


# In[86]:


from imblearn.over_sampling import RandomOverSampler
ros =  RandomOverSampler()


# In[87]:


x=df.drop(["is_legendary"],axis=1)


# In[88]:


y=df["is_legendary"]


# In[89]:


x_ros,y_ros=ros.fit_sample(x,y)


# In[90]:


x_ros.shape,y_ros.shape


# In[91]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_ros,y_ros,test_size=0.33,random_state=5)


# In[92]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[93]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()


# In[94]:


df.dtypes


# In[95]:


lr.fit(x_train,y_train)


# In[96]:


lr.score(x_test,y_test)


# In[98]:


y_pred=lr.predict(x_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(lr.score(x_test, y_test)))


# In[99]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test,y_pred)
print(confusion_matrix)


# In[100]:


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[101]:


from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier()


# In[102]:


abc.fit(x_train,y_train)


# In[103]:


abc.score(x_test,y_test)


# In[104]:


from sklearn.ensemble import BaggingClassifier
bc = BaggingClassifier()


# In[105]:


bc.fit(x_train,y_train)


# In[106]:


bc.score(x_test,y_test)


# In[108]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()


# In[109]:


rfc.fit(x_train,y_train)


# In[110]:


rfc.score(x_test,y_test)


# In[112]:





# In[ ]:





# In[ ]:




