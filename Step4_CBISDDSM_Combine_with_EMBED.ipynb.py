#!/usr/bin/env python
# coding: utf-8

# In[58]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
from pydicom import dcmread
from pydicom.data import get_testdata_file
from skimage.transform import resize
import os
import shutil

def read_meta_data(filename):
    dataset = pd.read_csv(filename)
    return dataset


# In[59]:


cbisddsm = read_meta_data("/data/CBIS-DDSM/manifest-ZkhPvrLo5216730872708713142/metadata.csv")


# In[60]:



cbisddsm.head(5)


# In[61]:


cbisddsm['File Location'][0]


# In[ ]:





# In[62]:


# import pandas as pd
# data = {'day': ['3-20-2019', None, '2-25-2019'] }
# df = pd.DataFrame( data )

# df['day'] = pd.to_datetime(df['day'])
# df['day'] = df['day'].dt.strftime('%d.%m.%Y')
# df[ df == 'NaT' ] = '' 
cbisddsm['Study Date'] = pd.to_datetime(cbisddsm['Study Date'])
cbisddsm['Study Date'] = cbisddsm['Study Date'].dt.strftime('%d.%m.%Y')
cbisddsm[cbisddsm['Study Date'] == 'NaT'] = ''
cbisddsm['Study Date'].sort_values(ascending=True)


# In[ ]:





# In[63]:


def MalignOrBenign(DistinctName):
    cal_case_train_set = read_meta_data("calc_case_description_train_set.csv")
    cal_case_test_set = read_meta_data("calc_case_description_test_set.csv")
    mass_case_train_set = read_meta_data("mass_case_description_test_set.csv")
    mass_case_test_set = read_meta_data("mass_case_description_train_set.csv")

    cal_case_train = cal_case_train_set[cal_case_train_set["image file path"].str.contains(DistinctName)]
    cal_case_test = cal_case_test_set[cal_case_test_set["image file path"].str.contains(DistinctName)]
    mass_case_train = mass_case_train_set[mass_case_train_set["image file path"].str.contains(DistinctName)]
    mass_case_test = mass_case_test_set[mass_case_test_set["image file path"].str.contains(DistinctName)]

    if cal_case_train.shape[0] > 0:
        label = cal_case_train['pathology'].values.astype('str')[0]
        return label

    if cal_case_test.shape[0] > 0:
        label = cal_case_test['pathology'].values.astype('str')[0]
        return label

    if mass_case_train.shape[0] > 0:
        label = mass_case_train['pathology'].values.astype('str')[0]
        return label

    if mass_case_test.shape[0] > 0:
        label = mass_case_test['pathology'].values.astype('str')[0]
        return label

    return "No_Label"


# In[64]:


def ConvertFromDCMtoPNG(srcPath, dstFolderPath):
    im_dim_x = 800
    im_dim_y = 600
    ## Read DCM files with Pydicom 
    ## All DCM files are converted into 16Bit pngs.
    im = pydicom.dcmread(srcPath)
    print(srcPath + "'s resolution is " + str(im.pixel_array.shape))
    im = im.pixel_array.astype(float)

    im = resize(im, (im_dim_x, im_dim_y), anti_aliasing=True)

    rescaled_image = (np.maximum(im, 0)/im.max())*65536

    final_image= np.uint16(rescaled_image)
    print(srcPath + "'s resolution rescaled to " + str(final_image.shape))


    final_image = Image.fromarray(final_image)
    final_image.save(dstFolderPath)


# In[ ]:





# In[ ]:





# In[ ]:





# In[65]:


PositiveCBISDF = []
NegativeCBISDF = []
## Labelling Image Files as positives and negatives. benign_without_callbacks are negatives. Other files are all positives.
def RegenerateUIDandPath(dataset):
    for idx in dataset.index:
        lastChar = dataset['Subject ID'][idx][-1]
        if lastChar.isnumeric():
            print("the last character is numeric. ANd is going to be continued!")
            continue
        
        distinctiveName = dataset['Subject ID'][idx]
        distinctiveUID = dataset['Series UID'][idx]
        distinctiveRow = dataset.iloc[idx]
        
        RawPath = dataset['File Location'][idx] + "//1-1.dcm"
        NewPathSrc = "//data//CBIS-DDSM//manifest-ZkhPvrLo5216730872708713142//" + RawPath[1:len(RawPath)]
        
        NewPositivePath = "/home/jupyter-ihwan28/breast_CBISDDSM/images/800x600/benign_negative/pos/"
        NewNegativePath = "/home/jupyter-ihwan28/breast_CBISDDSM/images/800x600/benign_negative/neg/"
        if not os.path.exists(NewPositivePath):
        
            os.makedirs(NewPositivePath)
            print("Positive path is created!")
            
        if not os.path.exists(NewNegativePath):
        
            os.makedirs(NewNegativePath)
            print("Negative path is created!")
        
        label = MalignOrBenign(distinctiveName)
        print(NewPathSrc + " is : " + label)
        if 'BENIGN_WITHOUT_CALLBACK' == label.upper():
            destFolderPath =  NewNegativePath + str(idx) + ".png"
            NegativeCBISDF.append([distinctiveUID, destFolderPath])
            print("Benign_withoutcall_back :" + str(destFolderPath))
            print('label :' + label)
            print("---------------------------------------")
        elif 'BENIGN' == label.upper():
            destFolderPath =  NewPositivePath + str(idx) + ".png"
            PositiveCBISDF.append([distinctiveUID, destFolderPath])
            print("Malignant but Benign: " + str(destFolderPath))
            print('label :' + label)
            print("---------------------------------------")
        else:
            destFolderPath =  NewPositivePath + str(idx) + ".png"
            PositiveCBISDF.append([distinctiveUID, destFolderPath])
            print("Malignant : " + str(destFolderPath))
            print('label :' + label)
            print("---------------------------------------")
        ConvertFromDCMtoPNG(NewPathSrc, destFolderPath)
        


# In[66]:


RegenerateUIDandPath(cbisddsm)


# In[67]:


## Column names are added appropriately
neg_df = pd.DataFrame(NegativeCBISDF, columns=['empi_anon', 'file_path'])
pos_df = pd.DataFrame(PositiveCBISDF, columns=['empi_anon', 'file_path'])


# In[68]:


print(len(neg_df))


# In[69]:


print(len(pos_df))


# In[70]:


## metadata are stored into CSV files. 

neg_df.to_csv('/home/jupyter-ihwan28/breast_CBISDDSM/images/800x600/benign_negative/neg_empi_path.csv')
pos_df.to_csv('/home/jupyter-ihwan28/breast_CBISDDSM/images/800x600/benign_negative/pos_empi_path.csv')


# In[ ]:




