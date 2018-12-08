
# coding: utf-8

# In[1]:


import os
import sys
import time
import datetime
import socket
import yaml
import shutil
import glob

import pandas as pd
import numpy as np

from sklearn.externals import joblib 

from sklearn.metrics import accuracy_score, log_loss, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#######################################################################
featureType = ['wtio', 'slbp', 'haralick']
kernel = 'db2'
level = '5'
data = 'melanoma_binary'
feaList = []
for i,ftp in enumerate(featureType):
	feaFolderList = []

	dirt = glob.glob('/home/akhiyar/Software/xprmt/feature/extract-'+featureType[i]+'-'+data+'-*')
	print(dirt)
	
	
	if ftp == 'wtio':
		for dirPath in dirt:
		
			cfgPath = os.path.join(dirPath, [f for f in os.listdir(dirPath) if f.endswith('.cfg')][0])
			print(cfgPath)
			dat = pd.read_fwf(cfgPath)
			ker = dat['Unnamed: 1'][0:1].values[0].split()[-1][1:-2]
			lvl = dat['Unnamed: 1'][1:2].values[0].split()[-1][0]
			#print(ker)
			#print(lvl)
			if ker == kernel and lvl == level:
				feaFolderList.append(os.path.join(dirPath,data))

	else:
		feaFolderList.append(os.path.join(dirt[0],data))

	if len(feaFolderList) > 1:
		feaFolderList = feaFolderList[0:1]

	feaList.extend(feaFolderList)

print("feature")
print(len(feaList))

#######################################################################
# In[2]:


# folderFeature = 'feature'
# feaPaths = ['../xprmt/'+folderFeature+'/extract-haralick-DataPH2_lesion_hairremove_augmented-mlbox-20181117-145539/DataPH2_lesion_hairremove_augmented',
#            '../xprmt/'+folderFeature+'/extract-slbp-DataPH2_lesion_hairremove_augmented-mlbox-20181117-144735/DataPH2_lesion_hairremove_augmented',
#            '../xprmt/'+folderFeature+'/extract-wtio-DataPH2_lesion_hairremove_augmented-mlbox-20181119-082251/DataPH2_lesion_hairremove_augmented',
#           ]
feaPaths = feaList


featureList = []
labelList = []
for feaPath in feaPaths:
    print(feaPath)
    xListL = []
    yListL = []
    yLi = os.listdir(feaPath)
    sumdat = 0
    for i in yLi:
        file = os.listdir(feaPath+ "/" + i)
        cnt = 0
        for j in file:
            if j.endswith('.pkl'):
                #print(j)
                x = joblib.load(feaPath + "/" + i + "/" +j)
                xListL.append(x)
                yListL.append(i)
                cnt += 1
        print(str(i) + " " + str(cnt))
        sumdat += cnt
    
    print("semuanya "+ str(sumdat))
    featureList.append(xListL)
    labelList.append(yListL)

xList = np.concatenate((featureList[0], featureList[1]), axis=1)
for idx in range(len(featureList)-2):
    xList = np.concatenate((xList, featureList[idx+2]), axis=1)

xList1 = xList
yList1 = labelList[0]


# In[3]:


xList = np.array(xList1)
yList = np.array(yList1)
print(xList.shape)
print(yList.shape)


# In[4]:


# from sklearn.preprocessing import normalize
# xList = normalize(xList, norm='l2', axis=0)


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(xList, yList, random_state = 42, test_size = 0.2)


# In[6]:


print(X_train.shape)


# In[7]:


from sklearn.ensemble import RandomForestClassifier
from imblearn.metrics import sensitivity_specificity_support
clf = RandomForestClassifier(random_state=6)
clf.fit(X_train, y_train)
prediksi = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediksi))
print(classification_report(y_test, prediksi))
print(sensitivity_specificity_support(y_test, prediksi, average='macro'))


# In[8]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
from sklearn.model_selection import cross_val_score
score = cross_val_score(clf, xList, yList, cv=10)
print(score)
print(score.mean())


# In[ ]:


clf1 = RandomForestClassifier(random_state=1)
clf2 = GradientBoostingClassifier()
clf3 = LinearDiscriminantAnalysis()
eclf1 = VotingClassifier(estimators=[('rf', clf1), ('gb', clf2), ('lda', clf3)], voting='hard')
score = cross_val_score(clf, xList, yList, cv=10)
print(score)
print(score.mean())


# In[ ]:


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", probability=True),
    #NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(random_state=1),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]

# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

for clf in classifiers:
    from sklearn.preprocessing import normalize
    X_train = normalize(X_train, norm='l2', axis=0)
    X_test = normalize(X_test, norm='l2', axis=0)

    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print("Log Loss: {}".format(ll))
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
    ######################### perform 10 fold validation ######################
    kf = StratifiedKFold(n_splits=10, random_state=42)
    scorelist = []
    sensitivity = []
    specificity = []
    f1 = []
    for train_index, test_index in kf.split(X_train, y_train):
        clf.fit(X_train[train_index], y_train[train_index])
        p = clf.predict(X_train[test_index])
        accval = accuracy_score(y_train[test_index], p)
        scorelist.append(accval)
        sss = sensitivity_specificity_support(y_train[test_index], p, average='macro')
        sensitivity.append(sss[0])
        specificity.append(sss[1])
        f1.append(f1_score(y_train[test_index], p, average='macro') )

        
    print("MeanCVScore: {}".format(sum(scorelist)/len(scorelist)))
    print("10FoldCVScore: {}".format(scorelist))
    print("sensitivity: {}".format(sum(sensitivity)/len(sensitivity)))
    print("specificity: {}".format(sum(specificity)/len(specificity)))
    print("f1-score: {}".format(sum(f1)/len(f1)))

    #############################################################################
    
print("="*30)


# In[ ]:


# clf1 = RandomForestClassifier(random_state=1)
# clf2 = GradientBoostingClassifier()
# clf3 = LinearDiscriminantAnalysis()
# eclf1 = VotingClassifier(estimators=[('rf', clf1), ('gb', clf2), ('lda', clf3)], voting='hard')
# score = cross_val_score(clf, xList, yList, cv=10)
# print(score)
# print(score.mean())


# In[ ]:


# cm1 = confusion_matrix(yList[test_index],p)
# print('Confusion Matrix : \n', cm1)

# total1=sum(sum(cm1))
# #####from confusion matrix calculate accuracy
# accuracy1=(cm1[0,0]+cm1[1,1])/total1
# print ('Accuracy : ', accuracy1)

# sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
# print('Sensitivity : ', sensitivity1 )

# specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
# print('Specificity : ', specificity1)

# print(classification_report(yList[test_index], p))
    
# score = cross_val_score(clf, xList, yList, cv=10)
# print("MeanCVScore: {}".format(score.mean()))
# print("10FoldCVScore: {}".format(score))


# In[ ]:


# feaPath = '../xprmt/feature/extract-slbp-melanoma_binary_augmented-mlbox-20181117-151136/melanoma_binary_augmented'

# xListL = []
# yListL = []
# yLi = os.listdir(feaPath)
# sumdat = 0
# for i in yLi:
#     file = os.listdir(feaPath+ "/" + i)
#     cnt = 0
#     for j in file:
#         if j.endswith('.pkl'):
#             #print(j)
#             x = joblib.load(feaPath + "/" + i + "/" +j)
#             xListL.append(x)
#             yListL.append(i)
#             cnt += 1
#     print(str(i) + " " + str(cnt))
#     sumdat += cnt
# print("semuanya "+ str(sumdat))
# feaPath = '../xprmt/feature/extract-haralick-melanoma_binary_augmented-mlbox-20181117-151626/melanoma_binary_augmented'

# xListH = []
# yListH = []
# yLi = os.listdir(feaPath)
# sumdat = 0
# for i in yLi:
#     file = os.listdir(feaPath+ "/" + i)
#     cnt = 0
#     for j in file:
#         if j.endswith('.pkl'):
#             #print(j)
#             x = joblib.load(feaPath + "/" + i + "/" +j)
#             xListH.append(x)
#             yListH.append(i)
#             cnt += 1
#     print(str(i) + " " + str(cnt))
#     sumdat += cnt
# print("semuanya "+ str(sumdat))
# feaPath = '../xprmt/feature/extract-wtio-melanoma_binary_augmented-mlbox-20181117-145744/melanoma_binary_augmented'

# xListW = []
# yListW = []
# yLi = os.listdir(feaPath)
# sumdat = 0
# for i in yLi:
#     file = os.listdir(feaPath+ "/" + i)
#     cnt = 0
#     for j in file:
#         if j.endswith('.pkl'):
#             #print(j)
#             x = joblib.load(feaPath + "/" + i + "/" +j)
#             xListW.append(x)
#             yListW.append(i)
#             cnt += 1
#     print(str(i) + " " + str(cnt))
#     sumdat += cnt
# print("semuanya "+ str(sumdat))
# xListL = np.array(xListL)
# xListH = np.array(xListH)
# xListW = np.array(xListW)
# xList1 = np.concatenate((xListL, xListH, xListW), axis=1)
# #xList = np.concatenate((xListL, xListW), axis=1)
# print(yListL == yListH)
# print(yListH == yListW)
# yList1 = yListL

