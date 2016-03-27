
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


# In[2]:

from sklearn import cross_validation, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc
import pylab as pl


# In[3]:

from pandas import read_csv, DataFrame, Series
data = read_csv('~/kaggle/santander/train.csv')


# In[23]:

#throw away features, where >= 80% of values is equal to median
cdata = data
for col in cdata.columns[1:-1]:
    ser = cdata[col]
    cnt = ser.value_counts()
    med = ser.median()
    if cnt[med]/len(ser) >= 0.60:
        cdata = cdata.drop([col], axis=1)
#cdata


# In[24]:

print(len(cdata.columns))


# In[25]:

#for col in cdata.columns[1:-1]:
#    cdata.pivot_table('ID', [col], 'TARGET', 'count').plot(title=col)


# In[26]:

Y = cdata['TARGET']
X = cdata.drop(['ID', 'TARGET'], axis=1)
kcv = 10
cvd = {}


# In[27]:

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[28]:

model_rfc = RandomForestClassifier(n_estimators = 20)
#model_knc = KNeighborsClassifier(n_neighbors = 18)
#model_lr = LogisticRegression(penalty='l2', tol=0.01) 
#model_svc = svm.SVC()


# In[29]:

rf_scores = cross_validation.cross_val_score(model_rfc, X_train, y_train, cv = kcv)
cvd['RandomForestClassifier'] = rf_scores.mean()


# In[30]:

#kn_scores = cross_validation.cross_val_score(model_knc, X_train, y_train, cv = kcv)
#cvd['KNeighborsClassifier'] = kn_scores.mean()


# In[31]:

#lr_scores = cross_validation.cross_val_score(model_lr, X_train, y_train, cv = kcv)
#cvd['LogisticRegression'] = lr_scores.mean()


# In[32]:

#svc_scores = cross_validation.cross_val_score(model_svc, X_train, y_train, cv = kcv)
#cvd['SVC'] = svc_scores.mean()


# In[33]:

print(rf_scores)
#print(kn_scores)
#print(lr_scores)

#DataFrame.from_dict(data = cvd, orient='index').plot(kind='bar', legend=False)
#plt.plot(range(kcv), rf_scores, 'r', range(kcv), kn_scores, 'g', range(kcv), lr_scores)#, 'b', range(kcv), svc_scores, 'c')


# In[34]:

pl.clf()

probas = model_rfc.fit(X_train, y_train).predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
roc_auc  = auc(fpr, tpr)
pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('RF',roc_auc))

#probas = model_knc.fit(X_train, y_train).predict_proba(X_test)
#fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
#roc_auc  = auc(fpr, tpr)
#pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('KN',roc_auc))

#probas = model_lr.fit(X_train, y_train).predict_proba(X_test)
#fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
#roc_auc  = auc(fpr, tpr)
#pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('LR',roc_auc))

#probas = model_svc.fit(X_train, y_train).predict_proba(X_test)
#fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
#roc_auc  = auc(fpr, tpr)
#pl.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % ('SVC',roc_auc))

pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.legend(loc=0, fontsize='small')
pl.show()


# In[ ]:



