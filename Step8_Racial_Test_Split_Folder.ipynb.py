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


### This racial dataset preparation is supposed to used for the testing phase. Testing the models over different races
##Racial information added metadata are read
negativeimg = pd.read_csv('/home/jupyter-ihwan28/breast_simple_comparison_by_race/images/800x600/br12_456/neg_empi_path.csv')
positiveimg = pd.read_csv('/home/jupyter-ihwan28/breast_simple_comparison_by_race/images/800x600/br12_456/pos_empi_path.csv')

negativeimg = shuffle(negativeimg, random_state=4)
positiveimg = shuffle(positiveimg, random_state=4)


# In[8]:


negativeimg.groupby(['race'])['race'].count()


# In[9]:


positiveimg.groupby(['race'])['race'].count()


# In[19]:


## negative datasets are split in terms of races.
NegativeBlack = negativeimg[negativeimg.race == 'African American  or Black']
NegativeWhite = negativeimg[negativeimg.race == 'Caucasian or White']
NegativeAsian = negativeimg[negativeimg.race == 'Asian']
NegativeUnknown = negativeimg[negativeimg.race == 'Unknown, Unavailable or Unreported']


# In[20]:


## Positive datasets are split in terms of races.
positiveimgBlack = positiveimg[positiveimg.race == 'African American  or Black']
positiveimgWhite = positiveimg[positiveimg.race == 'Caucasian or White']
positiveimgAsian = positiveimg[positiveimg.race == 'Asian']
positiveimgUnknown = positiveimg[positiveimg.race == 'Unknown, Unavailable or Unreported']


# In[25]:


len(positiveimgBlack)


# In[24]:


def TestSeperateFolder_Black(TestPositive, TestNegative, suffix = '_800_600_simple_comparison_Black'):


    if not os.path.exists('test' + suffix):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs("test" + suffix)
        os.makedirs("test" + suffix + "/pos/")
        os.makedirs("test" + suffix + "/neg/")


    PostiveCount = 0
    NegativeCount = 0

    for index in TestPositive.index:
        srcPath = TestPositive['file_path'][index]
        shutil.copy(srcPath, "./test"+ suffix+"/pos") 
        print(srcPath + " has been copied(TestPositive positive)")
        PostiveCount += 1
        print(PostiveCount)
    
    for index in TestNegative.index:
        srcPath = TestNegative['file_path'][index]
        shutil.copy(srcPath, "./test"+ suffix+"/neg") 
        print(srcPath + " has been copied(TestNegative negative)")
        NegativeCount += 1
        print(NegativeCount)



        
        


TestSeperateFolder_Black(positiveimgBlack, NegativeBlack)


# In[26]:


def TestSeperateFolder_Black(TestPositive, TestNegative, suffix = '_800_600_simple_comparison_White'):


    if not os.path.exists('test' + suffix):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs("test" + suffix)
        os.makedirs("test" + suffix + "/pos/")
        os.makedirs("test" + suffix + "/neg/")

    PostiveCount = 0
    NegativeCount = 0

    for index in TestPositive.index:
        srcPath = TestPositive['file_path'][index]
        shutil.copy(srcPath, "./test"+ suffix+"/pos") 
        print(srcPath + " has been copied(TestPositive positive)")
        PostiveCount += 1
        print(PostiveCount)
    
    for index in TestNegative.index:
        srcPath = TestNegative['file_path'][index]
        shutil.copy(srcPath, "./test"+ suffix+"/neg") 
        print(srcPath + " has been copied(TestNegative negative)")
        NegativeCount += 1
        print(NegativeCount)



        
        


TestSeperateFolder_Black(positiveimgWhite, NegativeWhite)


# In[27]:


def TestSeperateFolder_Black(TestPositive, TestNegative, suffix = '_800_600_simple_comparison_Asian'):


    if not os.path.exists('test' + suffix):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs("test" + suffix)
        os.makedirs("test" + suffix + "/pos/")
        os.makedirs("test" + suffix + "/neg/")

    PostiveCount = 0
    NegativeCount = 0

    for index in TestPositive.index:
        srcPath = TestPositive['file_path'][index]
        shutil.copy(srcPath, "./test"+ suffix+"/pos") 
        print(srcPath + " has been copied(TestPositive positive)")
        PostiveCount += 1
        print(PostiveCount)
    
    for index in TestNegative.index:
        srcPath = TestNegative['file_path'][index]
        shutil.copy(srcPath, "./test"+ suffix+"/neg") 
        print(srcPath + " has been copied(TestNegative negative)")
        NegativeCount += 1
        print(NegativeCount)



        
        


TestSeperateFolder_Black(positiveimgAsian, NegativeAsian)


# In[28]:


def TestSeperateFolder_Black(TestPositive, TestNegative, suffix = '_800_600_simple_comparison_unknown'):


    if not os.path.exists('test' + suffix):
        # if the demo_folder directory is not present
        # then create it.
        os.makedirs("test" + suffix)
        os.makedirs("test" + suffix + "/pos/")
        os.makedirs("test" + suffix + "/neg/")

    PostiveCount = 0
    NegativeCount = 0

    for index in TestPositive.index:
        srcPath = TestPositive['file_path'][index]
        shutil.copy(srcPath, "./test"+ suffix+"/pos") 
        print(srcPath + " has been copied(TestPositive positive)")
        PostiveCount += 1
        print(PostiveCount)
    
    for index in TestNegative.index:
        srcPath = TestNegative['file_path'][index]
        shutil.copy(srcPath, "./test"+ suffix+"/neg") 
        print(srcPath + " has been copied(TestNegative negative)")
        NegativeCount += 1
        print(NegativeCount)



        
        
# ROIRemovedMammoDF = read_meta_data()
# TrainMal, TestMal, ValMal, TrainNeg, TestNeg, ValNeg = SplitBIRAD1_others(negFileDf, posFileDf)


TestSeperateFolder_Black(positiveimgUnknown, NegativeUnknown)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




