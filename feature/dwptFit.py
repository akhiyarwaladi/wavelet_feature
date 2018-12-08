
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
kernel = 'coif1'
level = '4'
data = 'melanoma_hair'
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
feaPaths1 = feaList
#######################################################################


#######################################################################
featureType = ['wtio', 'slbp', 'haralick']
kernel = 'coif1'
level = '4'
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
feaPaths2= feaList
#######################################################################

# folderFeature = 'feature'
# feaPaths1 = ['../xprmt/'+folderFeature+'/extract-haralick-melanoma_hair-mlbox-20181119-170051/melanoma_hair',
#            '../xprmt/'+folderFeature+'/extract-slbp-melanoma_hair-mlbox-20181119-165721/melanoma_hair',
#            '../xprmt/'+folderFeature+'/extract-wtio-melanoma_hair-mlbox-20181119-145953/melanoma_hair',
#           ]

featureList = []
labelList = []
for feaPath in feaPaths1:
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


# folderFeature = 'feature'
# feaPaths2 = ['../xprmt/'+folderFeature+'/extract-haralick-melanoma_binary-mlbox-20181117-180342/melanoma_binary',
#            '../xprmt/'+folderFeature+'/extract-slbp-melanoma_binary-mlbox-20181117-175922/melanoma_binary',
#            '../xprmt/'+folderFeature+'/extract-wtio-melanoma_binary-mlbox-20181117-174743/melanoma_binary',
#           ]

featureList = []
labelList = []
for feaPath in feaPaths2:
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

xList2 = xList
yList2 = labelList[0]


# In[4]:


xList = np.concatenate((xList1, xList2), axis=0)
yList1.extend(yList2)
yList = yList1
print(len(yList))


# In[5]:


xList = np.array(xList)
yList = np.array(yList)
print(xList.shape)
print(yList.shape)


# In[6]:


from sklearn.preprocessing import normalize
xList = normalize(xList, norm='l2', axis=0)


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(xList, yList, random_state = 42, test_size = 0.2)


# In[8]:


print(X_train.shape)


# In[9]:


from sklearn.ensemble import RandomForestClassifier
from imblearn.metrics import sensitivity_specificity_support
clf = RandomForestClassifier(random_state=6)
clf.fit(X_train, y_train)
prediksi = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, prediksi))
print(classification_report(y_test, prediksi))
print(sensitivity_specificity_support(y_test, prediksi, average='macro'))


# In[10]:


clf1 = RandomForestClassifier(random_state=1)
clf2 = GradientBoostingClassifier()
clf3 = LinearDiscriminantAnalysis()
eclf1 = VotingClassifier(estimators=[('rf', clf1), ('gb', clf2), ('lda', clf3)], voting='hard')
score = cross_val_score(clf, xList, yList, cv=10)
print(score)
print(score.mean())


# In[11]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
from sklearn.model_selection import cross_val_score
score = cross_val_score(clf, xList, yList, cv=10)
print(score)
print(score.mean())


# In[12]:



classifiers = [
    KNeighborsClassifier(3),
    #SVC(kernel="rbf", C=0.025, probability=True),
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


# In[13]:


# feaPath = '../xprmt/feature1/extract-slbp-DataPH2-mlbox-20181116-021021/DataPH2'

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
# feaPath = '../xprmt/feature1/extract-haralick-DataPH2-mlbox-20181116-021704/DataPH2'

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
# feaPath = '../xprmt/feature1/extract-wtio-DataPH2-mlbox-20181116-021739/DataPH2'

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
# xList2 = np.concatenate((xListL, xListH, xListW), axis=1)
# #xList = np.concatenate((xListL, xListW), axis=1)
# print(yListL == yListH)
# print(yListH == yListW)
# yList2 = yListL

