#!/usr/bin/env python
# coding: utf-8

# ## Data Science Project 1 ##

# The data-set represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes
# 
# The main objective of the data-set is to record whether a patient would be readmitted within 30 days or after 30 days

# In[4]:


#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# In[5]:


#importing the dataset
diabetes = pd.read_csv(r'diabetic_data.csv')
diabetes.head(5)


# In[6]:


diabetes.head()


# In[7]:


#finding the null values
diabetes.isnull().sum()


# In[8]:


#to find the total count of the instances
diabetes.count()


# In[9]:


diabetes.describe()


# In[10]:


diabetes.info()


# In[11]:


#replacing the "?" values in the dataset using np.nan
diabetes = diabetes.replace("?",np.nan)


# In[12]:


#to find the null values after replacing it with nan
diabetes.isnull().sum()


# In[13]:


with pd.option_context("display.max_rows",None,"display.max_columns",None):
    print(diabetes.dtypes)


# In[14]:


#find the total no. of observations and variables
diabetes.shape


# In[15]:


diabetes.gender.value_counts()


# # MISSING VALUES TREATMENT

# In[16]:


#to drop columns related to ID'S
diabetes = diabetes.drop(['encounter_id','patient_nbr','admission_source_id'],axis=1)


# In[17]:


diabetes.shape


# In[18]:


#to drop columns having missing values more than 75%
diabetes = diabetes.drop(["weight","payer_code","medical_specialty"],axis = 1)


# In[19]:


#to drop this column due to attribute feature importance in the decision tree classifier
diabetes =  diabetes.drop(["diag_1"],axis=1)


# In[20]:


diabetes.shape


# In[21]:


diabetes.isnull().sum()


# In[22]:


diabetes.race.mode()
diabetes.diag_2.mode()
diabetes.diag_3.mode()


# In[23]:


#imputing the null values with mode
for value in ['race',"diag_2",'diag_3']:
    diabetes[value].fillna(diabetes[value].mode()[0])


# In[24]:


plt.figure(figsize=(12,8))
sb.boxplot(data=diabetes)
plt.xticks(rotation=90)


# In[25]:


# apply square root transformation on right skewed count data to reduce the effects of extreme values.
# here log transformation is not appropriate because the data is Poisson distributed and contains many zero values.
diabetes['number_outpatient'] = diabetes['number_outpatient'].apply(lambda x: np.sqrt(x + 0.5))
diabetes['number_emergency'] = diabetes['number_emergency'].apply(lambda x: np.sqrt(x + 0.5))
diabetes['number_inpatient'] = diabetes['number_inpatient'].apply(lambda x: np.sqrt(x + 0.5))


# In[26]:


#shows relation between one categorical and one numerical

sb.catplot(x = "gender",y =  "num_procedures",data=diabetes,hue =  "readmitted")
plt.figure(figsize=(15,12))


# In[27]:


#Show the counts of observations in each categorical bin using bars.
plt.figure(figsize=(10,6))
sb.countplot(x = "race",data=diabetes,hue="gender")


# In[28]:


plt.figure(figsize=(10,8))
sb.countplot(x="gender",hue="readmitted",data=diabetes)


# In[29]:


plt.figure(figsize=(10,6))
sb.stripplot(x = 'readmitted', y = "number_diagnoses",data=diabetes)
sb.set_style('whitegrid')


# In[30]:


plt.figure(figsize=(10,8))
sb.violinplot(x = "race",y="num_lab_procedures",data=diabetes)


# In[31]:


plt.hist(x = "num_lab_procedures",data=diabetes,bins=10)


# In[32]:


plt.hist(x = "num_medications",data=diabetes)


# In[33]:


diabetes.race.value_counts()


# In[34]:


diabetes.readmitted.value_counts()


# In[35]:


diabetes.groupby("age").size()


# In[36]:


age_id = {"[0-10)":0,
      "[10-20)":10,
      "[20-30)":20,
      "[30-40)":30,
      "[40-50)":40,
      "[50-60)":50,
      "[60-70)":60,
      "[70-80)":70,
      "[80-90)":80,
      "[90-100)":90}


# In[37]:


diabetes["age_group"] = diabetes.age.replace(age_id)


# In[38]:


sb.countplot(x = "age_group",data=diabetes)


# In[39]:


diabetes.head()


# In[40]:


diabetes = diabetes.drop(["age"],axis=1)


# In[41]:


diabetes.head()


# In[42]:


a  = diabetes.columns
a


# In[43]:



#diabetes['readmitted'] = pd.Series([1 if val == 'NO' else 0 for val in diabetes['readmitted']])
#diabetes.head()


# In[44]:


# original 'discharge_disposition_id' contains 28 levels
# reduce 'discharge_disposition_id' levels into 2 categories
# discharge_disposition_id = 1 corresponds to 'Discharge Home'
diabetes['discharge_disposition_id'] = pd.Series(['Home' if val == 1 else 'Other discharge' 
                                              for val in diabetes['discharge_disposition_id']], index=diabetes.index)


# In[45]:


# original 'admission_type_id' contains 8 levels
# reduce 'admission_type_id' into 2 categories
diabetes['admission_type_id'] = pd.Series(['Emergency' if val == 1 else 'Other type' 
                                              for val in diabetes['admission_type_id']], index=diabetes.index)


# In[46]:


diabetes.groupby('readmitted').size().plot(kind='bar')

plt.ylabel("Count")


# In[47]:


diabetes.dtypes


# In[48]:


diabetes.columns


# In[49]:


diabetes.dtypes


# In[50]:


#object dtype in single list
colname=[]
for b in diabetes.columns:
    if diabetes[b].dtype=='object':
        colname.append(b) 
        
colname


# In[51]:


diabetes.dtypes


# In[52]:


#for preprecessing and converting the object dtype into int type
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


for f in colname:
    diabetes[f] = le.fit_transform(diabetes[f].astype(str))


# In[53]:


sb.barplot(x = "gender",y="readmitted",data=diabetes)


# In[54]:


sb.countplot(x="number_inpatient",data=diabetes)
plt.xticks(rotation=90)


# In[55]:


sb.countplot(x="race",hue="gender",data=diabetes)


# In[56]:


sb.barplot(x="gender",y="number_inpatient",data=diabetes)


# In[57]:


num_sel= {"time_in_hospital","number_inpatient","number_outpatient"}


# In[58]:


fig,axes=plt.subplots(1,3,figsize=(18,6))
for i,t in enumerate(num_sel):
    sb.boxplot(y=t,x = "readmitted",data=diabetes,ax = axes[i])


# In[59]:


diabetes.dtypes


# In[60]:


cor_matrix = diabetes.corr().round(2)
cor_matrix


# In[61]:


#CORREALTION PLOT

plt.figure(figsize=(30,28))
sb.heatmap(cor_matrix,annot=True,vmax=1,center=0,cbar="PuBuGn",vmin=-1)


# In[62]:


diabetes.readmitted.value_counts()


# In[63]:


# compare diabetes medications 'miglitol', 'nateglinide' and 'acarbose' with 'insulin', as an example
fig = plt.figure(figsize=(20,18))

ax1 = fig.add_subplot(331)
ax1 = diabetes.groupby('miglitol').size().plot(kind='bar')
plt.xlabel('miglitol', fontsize=15)
plt.ylabel('Count', fontsize=15)

ax2 = fig.add_subplot(332)
ax2 = diabetes.groupby('nateglinide').size().plot(kind='bar')
plt.xlabel('nateglinide', fontsize=15)
plt.ylabel('Count', fontsize=15)

ax3 = fig.add_subplot(333)
ax3 = diabetes.groupby('acarbose').size().plot(kind='bar')
plt.xlabel('acarbose', fontsize=15)
plt.ylabel('Count', fontsize=15)

ax4 = fig.add_subplot(334)
ax4 = diabetes.groupby('insulin').size().plot(kind='bar')
plt.xlabel('insulin', fontsize=15)
plt.ylabel('Count', fontsize=15)

ax5 = fig.add_subplot(335)
ax5 = diabetes.groupby('rosiglitazone').size().plot(kind='bar')
plt.xlabel('rosiglitazone', fontsize=15)
plt.ylabel('Count', fontsize=15)

ax6 = fig.add_subplot(336)
ax6 = diabetes.groupby('glyburide-metformin').size().plot(kind='bar')
plt.xlabel('glyburide-metformin', fontsize=15)
plt.ylabel('Count', fontsize=15)




# In[64]:


#depending on feature_importance of dtc(decision tree classifier object) drop the variables
diabetes = diabetes.drop(['troglitazone','metformin-rosiglitazone','examide','citoglipton','glipizide-metformin','glimepiride-pioglitazone','metformin-rosiglitazone','metformin-pioglitazone','miglitol','acetohexamide','metformin','nateglinide','glyburide',
                          'glyburide-metformin','change'],axis=1)


# In[65]:


diabetes.shape


# In[66]:


features = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 
            'number_diagnoses', 'number_inpatient', 'number_emergency', 'number_outpatient']


# In[67]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler().fit_transform(diabetes[features])

diabetes_df = pd.DataFrame(data=scalar, columns=features, index=diabetes.index)
diabetes.drop(features, axis=1,inplace=True)
diabetes = pd.concat([diabetes, diabetes_df], axis=1)


# In[68]:


x = diabetes.drop(["readmitted"],axis=1)


# In[69]:


y = diabetes["readmitted"]


# In[70]:


# split X and y into cross-validation (75%) and testing (25%) data sets
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# In[71]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[72]:


from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix


# In[126]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=50,penalty="l2")


# In[74]:


lr.fit(x_train,y_train)


# In[75]:


from sklearn import metrics
y_pred = lr.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))


# In[76]:


lr.score(x_test,y_test)


# In[77]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc_score = cross_val_score(rfc, x_train, y_train, cv=5, scoring='accuracy').mean()
rfc_score


# In[78]:


rfc.fit(x_train,y_train)


# In[79]:


print(list(zip(diabetes.columns,rfc.feature_importances_)))


# In[80]:


rfc.score(x_test,y_test)


# In[81]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc_score = cross_val_score(dtc, x_train, y_train, cv=5, scoring='accuracy').mean()
dtc_score


# In[82]:


dtc.fit(x_train,y_train)


# In[83]:


print(list(zip(diabetes.columns,dtc.feature_importances_)))


# In[84]:


dtc.score(x_test,y_test)


# In[85]:


from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier()
abc_score = cross_val_score(abc, x_train, y_train, cv=10, scoring='accuracy').mean()
abc_score


# In[86]:


abc.fit(x_train,y_train)


# In[87]:


abc.score(x_test,y_test)


# In[88]:


from sklearn.ensemble import BaggingClassifier
bc = BaggingClassifier()
bc_score = cross_val_score(bc, x_train, y_train, cv=5, scoring='accuracy').mean()
bc_score


# In[89]:


bc.fit(x_train,y_train)


# In[90]:


bc.score(x_test,y_test)


# In[91]:


from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc_score = cross_val_score(gbc, x_train, y_train, cv=5, scoring='accuracy').mean()
gbc_score


# In[92]:


gbc.fit(x_train,y_train)


# In[93]:


gbc.score(x_test,y_test)


# In[94]:


from sklearn.ensemble import ExtraTreesClassifier
etc = ExtraTreesClassifier()


# In[95]:


etc.fit(x_train,y_train)


# In[96]:


etc.score(x_test,y_test)


# In[97]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()


# In[98]:


mlp.fit(x_train,y_train)


# In[99]:


mlp.score(x_test,y_test)


# In[100]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(x_train, y_train)


# In[101]:


y_pred = knn.predict(x_test)
y_pred1  = rfc.predict(x_test)
y_pred2 = abc.predict(x_test)
y_pred3 = mlp.predict(x_test)
y_pred4 = dtc.predict(x_test)
y_pred5 = lr.predict(x_test)
y_pred6 = etc.predict(x_test)


# In[102]:


knn.score(x_test,y_test)


# In[103]:


#Table that describes the performance of a classification model.

#True Positives (TP): we correctly predicted that they do have diabetes

#True Negatives (TN): we correctly predicted that they don't have diabetes

#False Positives (FP): we incorrectly predicted that they do have diabetes (a "Type I error")

#False Negatives (FN): we incorrectly predicted that they don't have diabetes (a "Type II error")
#confusion matrix for knn classification
abc = confusion_matrix(y_test, y_pred)
abc


# In[104]:


#confusion matrix for rfc classifier
confusion_matrix2 = confusion_matrix(y_test,y_pred1)
confusion_matrix2


# In[105]:


#confusionmatrix for adaboost classifier
confusion_matrix3 = confusion_matrix(y_test, y_pred2)
confusion_matrix3


# In[106]:


#confusion matrix for mlp classifier
confusion_matrix4 = confusion_matrix(y_test, y_pred3)
confusion_matrix4


# In[107]:


#confusion matrix  for dtc classifer
confusion_matrix5 = confusion_matrix(y_test, y_pred4)
confusion_matrix5


# In[108]:


#confusion matrix for lr classifier
confusion_matrix6= confusion_matrix(y_test, y_pred5)
confusion_matrix6


# In[109]:


#confusion matrix for etc classifier
confusion_matrix7 = confusion_matrix(y_test, y_pred6)
confusion_matrix7


# In[110]:


from numpy import mean
from numpy import std
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=10)
model = DecisionTreeClassifier()
pipeline = Pipeline(steps=[('s',rfe),('m',model)])
# evaluate model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(pipeline, x, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# In[111]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[112]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred1))


# In[113]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred2))


# In[114]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred4))


# In[115]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred5))


# In[116]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred6))


# In[117]:


print (metrics.precision_score(y_test, y_pred))


# In[118]:


print (metrics.f1_score(y_test, y_pred))


# In[119]:


# store the predicted probabilities for class 1
y_pred_prob = lr.predict_proba(x_test)[:,1]


# In[120]:


y_score =rfc.predict_proba(x_test)[:,1]


# In[121]:


y_score1 = etc.predict_proba(x_test)[:,1]


# In[122]:


y_score2 = bc.predict_proba(x_test)[:,1]


# In[123]:


y_score3 = dtc.predict_proba(x_test)[:,1]


# In[124]:


fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_prob)
fpr_rfc, tpr_rfc, thresholds = metrics.roc_curve(y_test, y_score)
fpr_etc, tpr_etc, thresholds = metrics.roc_curve(y_test, y_score1)
fpr_bc, tpr_bc, thresholds = metrics.roc_curve(y_test, y_score2)
fpr_dtc, tpr_dtc, thresholds = metrics.roc_curve(y_test, y_score3)
plt.plot(fpr_bc, tpr_bc, label='extra trees classifier')
plt.plot(fpr_etc, tpr_etc, label='Randomforest Classifier')                                        
plt.plot(fpr, tpr,label = "bagging classifier")
plt.plot(fpr, tpr,label = "decision tree classifier")
plt.plot(fpr, tpr,label = "logistic regression")
plt.legend()
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title('ROC curve for diabetes readmission')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)


# In[125]:


# AUC score
print (metrics.roc_auc_score(y_test, y_pred_prob))


# In[ ]:




