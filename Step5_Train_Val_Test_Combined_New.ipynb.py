#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from dataclasses import dataclass
import os
import random
# from numba import njit, jit
import shutil
from sklearn.utils import shuffle


# In[2]:


## read csv files that has been generated from the previous steps.
negativeimg = pd.read_csv('/home/jupyter-ihwan28/breast_simple_comparison/images/800x600/br12_456/neg_empi_path.csv')
positiveimg = pd.read_csv('/home/jupyter-ihwan28/breast_simple_comparison/images/800x600/br12_456/pos_empi_path.csv')

negativeimg = shuffle(negativeimg, random_state=4)
positiveimg = shuffle(positiveimg, random_state=4)


# In[3]:


## shuffle image data randomly.
empi_anon_pos = shuffle(list(positiveimg['empi_anon'].unique()), random_state=4)
empi_anon_neg = shuffle(list(negativeimg['empi_anon'].unique()), random_state=4)
TrainSize = 0.6 * len(positiveimg)
TestValSize = 0.2 * len(positiveimg) 
print(len(empi_anon_pos))
patient_id_df_pos = pd.DataFrame(empi_anon_pos, columns=['empi_anon'])
patient_id_df_neg = pd.DataFrame(empi_anon_neg, columns=['empi_anon'])


# In[4]:


## Split dataset into Train, validation, and test positive sets while avoiding patient leakage between datasets.

trainset_pos = []
testset_pos = []
valset_pos = []
train_empi_anon_pos = []
test_empi_anon_pos = []
val_empi_anon_pos = []
for idx, anon in enumerate(empi_anon_pos):
    if idx == 0:
        trainset_pos = positiveimg[positiveimg['empi_anon'] == anon]
        
    
    if len(trainset_pos)>= TrainSize:
        print("Train : " + str(len(trainset_pos)))
        print("Break out")
        train_empi_anon_pos = [*set(train_empi_anon_pos)]
        break
        
    temp = positiveimg[positiveimg['empi_anon'] == anon]
    trainset_pos = trainset_pos.append(temp, ignore_index = True)
    train_empi_anon_pos.append(anon)
    
PatientID_Val_Test = patient_id_df_pos[~patient_id_df_pos.empi_anon.isin(train_empi_anon_pos)]

empi_anon_val = shuffle(list(PatientID_Val_Test['empi_anon'].unique()), random_state=4)
for idx, anon in enumerate(empi_anon_val):
    if idx == 0:
        valset_pos = positiveimg[positiveimg['empi_anon'] == anon]
        
    
    if len(valset_pos)>= TestValSize:
        print("Val : "+str(len(valset_pos)))
        print("Val Break out")
        val_empi_anon_pos = [*set(val_empi_anon_pos)]
        break
        
    temp = positiveimg[positiveimg['empi_anon'] == anon]
    valset_pos = valset_pos.append(temp, ignore_index = True)
    val_empi_anon_pos.append(anon)

PatientID_Test = PatientID_Val_Test[~PatientID_Val_Test.empi_anon.isin(val_empi_anon_pos)]

empi_anon_test = shuffle(list(PatientID_Test['empi_anon'].unique()), random_state=4)

for idx, anon in enumerate(empi_anon_test):
    if idx == 0:
        testset_pos = positiveimg[positiveimg['empi_anon'] == anon]

    temp = positiveimg[positiveimg['empi_anon'] == anon]
    testset_pos = testset_pos.append(temp, ignore_index = True)
    test_empi_anon_pos.append(anon)
    
    if idx == (len(empi_anon_test) - 1):
        print("Test : "+str(len(testset_pos)))
        test_empi_anon_pos = [*set(test_empi_anon_pos)]


# In[5]:


#### Split dataset into Train, validation, and test negative sets while avoiding patient leakage between datasets.
trainset_neg = []
testset_neg= []
valset_neg = []
train_empi_anon_neg = []
test_empi_anon_neg = []
val_empi_anon_neg = []
for idx, anon in enumerate(empi_anon_neg):
    if idx == 0:
        trainset_neg = negativeimg[negativeimg['empi_anon'] == anon]
        
    
    if len(trainset_neg)>= TrainSize:
        print("Train : " + str(len(trainset_neg)))
        print("Break out")
        train_empi_anon_neg = [*set(train_empi_anon_neg)]
        break
        
    temp = negativeimg[negativeimg['empi_anon'] == anon]
    trainset_neg = trainset_neg.append(temp, ignore_index = True)
    train_empi_anon_neg.append(anon)
    
PatientID_Val_Test = patient_id_df_neg[~patient_id_df_neg.empi_anon.isin(train_empi_anon_neg)]

empi_anon_val = shuffle(list(PatientID_Val_Test['empi_anon'].unique()), random_state=4)
for idx, anon in enumerate(empi_anon_val):
    if idx == 0:
        valset_neg = negativeimg[negativeimg['empi_anon'] == anon]
        
    
    if len(valset_neg)>= TestValSize:
        print("Val : "+str(len(valset_neg)))
        print("Val Break out")
        val_empi_anon_neg = [*set(val_empi_anon_neg)]
        break
        
    temp = negativeimg[negativeimg['empi_anon'] == anon]
    valset_neg = valset_neg.append(temp, ignore_index = True)
    val_empi_anon_neg.append(anon)

PatientID_Test = PatientID_Val_Test[~PatientID_Val_Test.empi_anon.isin(val_empi_anon_neg)]

empi_anon_test = shuffle(list(PatientID_Test['empi_anon'].unique()), random_state=4)

for idx, anon in enumerate(empi_anon_test):
    if idx == 0:
        testset_neg = negativeimg[negativeimg['empi_anon'] == anon]

    if len(testset_neg)>= TestValSize:
        print("Test : "+str(len(testset_neg)))
        print("Test Break out")
        test_empi_anon_neg = [*set(test_empi_anon_neg)]
        break
        
    temp = negativeimg[negativeimg['empi_anon'] == anon]
    testset_neg = testset_neg.append(temp, ignore_index = True)
    test_empi_anon_neg.append(anon)


# In[6]:


### distributing train, test, and validation sets into different folders.
def SplitJPGtoSeperateFolder(TrainMal, TestMal, ValMal, TrainNeg, TestNeg, ValNeg, suffix = '_800_600_combined_embed_cbis'):


    if not os.path.exists('..//breast_chan//train' + suffix):

        os.makedirs("..//breast_chan//train" + suffix)
        os.makedirs("..//breast_chan//train" + suffix + "/pos/")
        os.makedirs("..//breast_chan//train" + suffix + "/neg/")

    if not os.path.exists("..//breast_chan//test" + suffix):
        os.makedirs("..//breast_chan//test" + suffix)
        os.makedirs("..//breast_chan//test"+ suffix+"/pos/")
        os.makedirs("..//breast_chan//test"+ suffix+"/neg/")

    if not os.path.exists("..//breast_chan//val" + suffix):
        os.makedirs("..//breast_chan//val" + suffix)
        os.makedirs("..//breast_chan//val"+ suffix+"/pos/")
        os.makedirs("..//breast_chan//val"+ suffix+"/neg/")

    for index in TrainMal.index:
        srcPath = TrainMal['file_path'][index]
        shutil.copy(srcPath, "..//breast_chan//train"+ suffix+"/pos") 
        print(srcPath + " has been copied(train positive)")



    for index in TestMal.index:
        srcPath = TestMal['file_path'][index]
        shutil.copy(srcPath, '..//breast_chan//test'+ suffix+'/pos')
        print(srcPath + " has been copied(test positive)")

    for index in ValMal.index:
        srcPath = ValMal['file_path'][index]
        shutil.copy(srcPath, '..//breast_chan//val'+ suffix+'/pos')
        print(srcPath + ' has been copied(val positive)')

    for index in TrainNeg.index:
        srcPath = TrainNeg['file_path'][index]
        shutil.copy(srcPath, '..//breast_chan//train'+ suffix+'/neg')
        print(srcPath + ' has been copied(train negative)')

    for index in TestNeg.index:
        srcPath = TestNeg['file_path'][index]
        shutil.copy(srcPath, '..//breast_chan//test'+ suffix+'/neg')
        print(srcPath + ' has been copied(test negative)')

    for index in ValNeg.index:
        srcPath = ValNeg['file_path'][index]
        shutil.copy(srcPath, '..//breast_chan//val'+ suffix+'/neg')
        print(srcPath + ' has been copied(validation negative)')
        
        


SplitJPGtoSeperateFolder(trainset_pos, testset_pos, valset_pos, trainset_neg, testset_neg, valset_neg)


# In[7]:


## read csv files of CBIS-DDSM benign and negative sets. 
cbisnegativeimg = pd.read_csv('/home/jupyter-ihwan28/breast_CBISDDSM/images/800x600/benign_negative/neg_empi_path.csv')
cbispositiveimg = pd.read_csv('/home/jupyter-ihwan28/breast_CBISDDSM/images/800x600/benign_negative/pos_empi_path.csv')

cbisnegativeimg = shuffle(cbisnegativeimg, random_state=4)
cbispositiveimg = shuffle(cbispositiveimg, random_state=4)


# In[8]:


print(len(cbispositiveimg))
print(len(cbisnegativeimg))


# In[9]:


## Copy and mix all those CBIS-DDSM positive images to EMBED dataset.  
def SplitJPGtoSeperateFolderNoTest(TrainMal, TestMal, ValMal, TrainNeg, TestNeg, ValNeg, suffix = '_800_600_combined_embed_cbis'):


    if not os.path.exists('..//breast_chan//train' + suffix):
        os.makedirs("..//breast_chan//train" + suffix)
        os.makedirs("..//breast_chan//train" + suffix + "/pos/")
        os.makedirs("..//breast_chan//train" + suffix + "/neg/")

    if not os.path.exists("..//breast_chan//test" + suffix):
        os.makedirs("..//breast_chan//test" + suffix)
        os.makedirs("..//breast_chan//test"+ suffix+"/pos/")
        os.makedirs("..//breast_chan//test"+ suffix+"/neg/")

    if not os.path.exists("..//breast_chan//val" + suffix):
        os.makedirs("..//breast_chan//val" + suffix)
        os.makedirs("..//breast_chan//val"+ suffix+"/pos/")
        os.makedirs("..//breast_chan//val"+ suffix+"/neg/")

    for index in TrainMal.index:
        srcPath = TrainMal['file_path'][index]
        shutil.copy(srcPath, "..//breast_chan//train"+ suffix+"/pos") 
        print(srcPath + " has been copied(train positive)")




    for index in ValMal.index:
        srcPath = ValMal['file_path'][index]
        shutil.copy(srcPath, '..//breast_chan//val'+ suffix+'/pos')
        print(srcPath + ' has been copied(val positive)')

    for index in TrainNeg.index:
        srcPath = TrainNeg['file_path'][index]
        shutil.copy(srcPath, '..//breast_chan//train'+ suffix+'/neg')
        print(srcPath + ' has been copied(train positive)')



    for index in ValNeg.index:
        srcPath = ValNeg['file_path'][index]
        shutil.copy(srcPath, '..//breast_chan//val'+ suffix+'/neg')
        print(srcPath + ' has been copied(validation positive)')

        


# In[10]:


## Copy and mix all those CBIS-DDSM negative images to EMBED dataset.  
def SplitMALIGNMENT_BENIGN(neg_dataset, pos_dataset):


    trainMal = pos_dataset.sample(frac=0.6, random_state=200)
    testMal = pos_dataset.drop(trainMal.index)
    valMal = testMal.sample(frac = 0.5, random_state = 200)
    testMal = testMal.drop(valMal.index)

    trainOthers = neg_dataset.sample(frac = 0.6, random_state = 200)
    testOthers = neg_dataset.drop(trainOthers.index)
    valOthers = testOthers.sample(frac = 0.5, random_state = 200)
    testOthers = testOthers.drop(valOthers.index)

    return trainMal, testMal, valMal, trainOthers, testOthers, valOthers

trainMal, testMal, valMal, trainOthers, testOthers, valOthers = SplitMALIGNMENT_BENIGN(cbisnegativeimg, cbispositiveimg)
SplitJPGtoSeperateFolderNoTest(trainMal, testMal, valMal, trainOthers, testOthers, valOthers)


# In[11]:


## PNG converted DDSM image information is read and  labelled
def AllFindFiles():
    ListLJPEGPath = []
    for root, dirs, files in os.walk("./negatives"):
        for file in files:
            if file.endswith(".png"):
                apath = os.path.join(root, file)
                ListLJPEGPath.append(apath)
    return ListLJPEGPath


# In[12]:


## file path labels are labelled in one column
negativeFiles = AllFindFiles()
neg_df = pd.DataFrame(negativeFiles, columns=['file_path'])
print(neg_df)
train = len(neg_df) * 0.6
val = len(neg_df) * 0.2


# In[13]:


## Split DDSM into train, test, validation sets.but test will not be used.
def SplitBENIGN_Only(neg_dataset):

    trainOthers = neg_dataset.sample(frac = 0.6, random_state = 200)
    testOthers = neg_dataset.drop(trainOthers.index)
    valOthers = testOthers.sample(frac = 0.5, random_state = 200)
    testOthers = testOthers.drop(valOthers.index)

    return trainOthers, testOthers, valOthers


# In[14]:


## Split DDSM and add them into Train and validation sets.
def SplitJPGtoSeperateFolderNegativeOnlyNoTest(TrainNeg, TestNeg, ValNeg, suffix = '_800_600_combined_embed_cbis'):


    if not os.path.exists('..//breast_chan//train' + suffix):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs("..//breast_chan//train" + suffix)
        os.makedirs("..//breast_chan//train" + suffix + "/pos/")
        os.makedirs("..//breast_chan//train" + suffix + "/neg/")

    if not os.path.exists("..//breast_chan//test" + suffix):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs("..//breast_chan//test" + suffix)
        os.makedirs("..//breast_chan//test"+ suffix+"/pos/")
        os.makedirs("..//breast_chan//test"+ suffix+"/neg/")

    if not os.path.exists("..//breast_chan//val" + suffix):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs("..//breast_chan//val" + suffix)
        os.makedirs("..//breast_chan//val"+ suffix+"/pos/")
        os.makedirs("..//breast_chan//val"+ suffix+"/neg/")


    for index in TrainNeg.index:
        srcPath = TrainNeg['file_path'][index]
        shutil.copy(srcPath, '..//breast_chan//train'+ suffix+'/neg')
        print(srcPath + ' has been copied(train negative)')



    for index in ValNeg.index:
        srcPath = ValNeg['file_path'][index]
        shutil.copy(srcPath, '..//breast_chan//val'+ suffix+'/neg')
        print(srcPath + ' has been copied(validation negative)')


# In[15]:


trainOthers, testOthers, valOthers = SplitBENIGN_Only(neg_df)
SplitJPGtoSeperateFolderNegativeOnlyNoTest(trainOthers, testOthers, valOthers)


# In[ ]:




