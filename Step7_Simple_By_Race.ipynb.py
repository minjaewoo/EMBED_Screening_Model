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
from numba import njit, jit
from sklearn.utils import shuffle
import shutil


# In[2]:


## Read EMBED dataset metadata files.
magview_path = '/data/mammo/png/magview_all_cohorts_anon_HITI.csv'
metadata_path = '/data/mammo/png/metadata_all_cohort_with_ROI_HITI.csv'


# In[ ]:





# In[3]:

## Read magview image findings file.
df_mag = pd.read_csv(magview_path)
df_mag['study_date_anon'] = pd.to_datetime(df_mag['study_date_anon'], errors='coerce', format= '%Y-%m-%d')
list(df_mag.columns)


# In[9]:


# see racial distribution.
df_mag.groupby(['ETHNICITY_DESC'])['ETHNICITY_DESC'].count()


# In[12]:


# df_mag['GENDER_DESC'].frequency()
df_mag.groupby(['GENDER_DESC'])['GENDER_DESC'].count()


# In[13]:


## Read file metadata files
df_meta = pd.read_csv(metadata_path)
list(df_meta.columns)


# In[14]:



## Data integrity check. EMBED only contains cohort 1 and 2.

print(df_meta.cohort_num.unique())
print(df_mag.cohort_num.unique())

df_meta.cohort_num = df_meta.cohort_num.astype(str) 
df_mag.cohort_num = df_mag.cohort_num.astype(str)



df_meta_one_two = df_meta[df_meta.cohort_num.isin(['1','2'])]
df_mag_one_two = df_mag[df_mag.cohort_num.isin(['1','2'])]


# In[15]:


print(df_meta_one_two.cohort_num.unique())
print(df_mag_one_two.cohort_num.unique())


# In[11]:


# Get diagnostic exams with BIRADS 4/5/6
df_mag_diag_pos = df_mag_one_two[df_mag_one_two.asses.isin(['S','M','K']) & 
                         df_mag_one_two.desc.str.contains('diag', case=False)]

# Get and rename relevant columns to prepare for merge with screening exams
## racial information is included in the magview file. 
df_mag_diag_pos_empi = df_mag_diag_pos[['empi_anon', 
                                        'acc_anon', 
                                        'numfind', 
                                        'bside', 
                                        'study_date_anon', 'ETHNICITY_DESC', 'GENDER_DESC', 'MARITAL_STATUS_DESC',
                                        'asses']]
## column names are labelled in simple language. 
df_mag_diag_pos_empi.columns = ['empi_anon', 
                                'acc_anon_diag', 
                                'diag_num', 
                                'diag_side', 
                                'diag_study_date', 'race', 'gender', 'marriage',
                                'diag_asses']

# Get screening exams and left merge on empi_anon with df_mag_diag_pos_empi
df_mag_scr = df_mag_one_two[df_mag_one_two.desc.str.contains('screen', case=False)]
df_mag_scr_pos = df_mag_diag_pos_empi.merge(df_mag_scr, on='empi_anon', how='left')

# Keep only screening exams with time diff less than +180 days
df_mag_scr_pos = df_mag_scr_pos.loc[(df_mag_scr_pos.side == df_mag_scr_pos.diag_side)]
df_mag_scr_pos['study_date_diff'] = df_mag_scr_pos.diag_study_date - df_mag_scr_pos.study_date_anon

df_mag_scr_pos_rel = df_mag_scr_pos.loc[(df_mag_scr_pos.study_date_diff.dt.days >= 0) & 
                                        (df_mag_scr_pos.study_date_diff.dt.days <= 180)]

# Get relevant columns from df_meta or merge entire dataset
df_meta_rel = df_meta_one_two[['empi_anon', 
                       'acc_anon', 
                       'ViewPosition', 
                       'ImageLateralityFinal', 
                       'FinalImageType', 
                       'png_path', 
                       'num_roi', 
                       'ROI_coords']]

# Merge df_meta_rel with positive exams in magview
df_meta_scr_pos_rel = df_mag_scr_pos_rel.merge(df_meta_rel, 
                                               left_on=['empi_anon', 'acc_anon', 'diag_side'], 
                                               right_on=['empi_anon', 'acc_anon', 'ImageLateralityFinal'], 
                                               how='inner')

# Keep only images with 1 or 2 ROIs
df_meta_scr_pos_rel = df_meta_scr_pos_rel[(df_meta_scr_pos_rel.num_roi == 1) | 
                                          (df_meta_scr_pos_rel.num_roi == 2)]

# Keep only 2D images
df_meta_scr_pos_rel = df_meta_scr_pos_rel[df_meta_scr_pos_rel.FinalImageType == '2D'].reset_index()

print(df_meta_scr_pos_rel.num_roi.value_counts())
print("Positive number")
print(len(df_meta_scr_pos_rel))


# In[16]:


## Eventual racial distribution displayed
df_meta_scr_pos_rel.groupby(['race'])['race'].count()


# In[17]:


df_meta_scr_pos_rel.groupby(['GENDER_DESC'])['GENDER_DESC'].count()


# In[ ]:





# In[18]:


## images are converted into 800 x 600 to make them fit into 8 GB GPUs.
import cv2 as cv

df = df_meta_scr_pos_rel
df = shuffle(df)
all_rois_list = []
all_file_empi_anon_pos = []
save_dir = '/home/jupyter-ihwan28/breast_simple_comparison_by_race/images/800x600/br12_456/'

def CoppFileToDirectory_Pos(Save_Dir):
    for i in range(len(df)):
        
        img_path = df.png_path[i]
        empi_anon = df.empi_anon[i]
        race = df.race[i]
        marriage = df.marriage[i]
        image_array = cv.imread(img_path)
        resized_image = cv.resize(image_array, (600, 800), interpolation= cv.INTER_AREA)
        if not os.path.exists(Save_Dir):
        # if the demo_folder directory is not present
        # then create it.
            os.makedirs(Save_Dir)
            os.makedirs(Save_Dir + "/pos/")
            os.makedirs(Save_Dir + "/neg/")
        print(str(i) + "th iteration")
        filename = os.path.basename(img_path)
        filename = filename +'_' +str(i) + '.png'
        newPath = Save_Dir+'/pos/' + filename
#         shutil.copy(img_path, newPath)
        cv.imwrite(newPath, resized_image)
        all_file_empi_anon_pos.append([empi_anon, newPath, race, marriage])

CoppFileToDirectory_Pos(save_dir)
## Information extracted are stored in separate csv files. 
pos_df = pd.DataFrame(all_file_empi_anon_pos, columns=['empi_anon', 'file_path', 'race', 'marriage'])
pos_df.to_csv('/home/jupyter-ihwan28/breast_simple_comparison_by_race/images/800x600/br12_456/pos_empi_path.csv')


# In[19]:


### Same processes are conducted for negative labelled images.

df_mag_scr_neg = df_mag_one_two[df_mag_one_two.desc.str.contains('screen', case=False) & 
                        df_mag_one_two.asses.isin(['N','B'])]

# Exclude patients that are included in the positive set
df_mag_scr_neg_rel = df_mag_scr_neg[~df_mag_scr_neg.empi_anon.isin(df_meta_scr_pos_rel.empi_anon)].sort_index()

# Merge df_meta_rel with negative screening exams in magview
df_meta_scr_neg_rel = pd.merge(df_mag_scr_neg_rel, 
                               df_meta_rel,
                               left_on=['empi_anon', 'acc_anon', 'side'], 
                               right_on=['empi_anon', 'acc_anon', 'ImageLateralityFinal'], 
                               how='inner')

# Keep only 2D images
df_meta_scr_neg_rel = df_meta_scr_neg_rel[df_meta_scr_neg_rel.FinalImageType == '2D'].reset_index()

# df_meta_scr_neg_rel
print("Negative number")
print(len(df_meta_scr_neg_rel))

# In[ ]:


df = df_meta_scr_neg_rel
df = shuffle(df)
all_rois_list = []
all_file_empi_anon_neg = []
save_dir = '/home/jupyter-ihwan28/breast_simple_comparison_by_race/images/800x600/br12_456/'

def CoppFileToDirectory_Neg(Save_Dir):
    for i in range(len(df)):
        print(str(i) + "th iteration")
        img_path = df.png_path[i]
        empi_anon = df.empi_anon[i]
        race = df.ETHNICITY_DESC[i]
        marriage = df.MARITAL_STATUS_DESC[i]
        image_array = cv.imread(img_path)
        resized_image = cv.resize(image_array, (600, 800), interpolation= cv.INTER_AREA)
        if not os.path.exists(Save_Dir):
        # if the demo_folder directory is not present
        # then create it.
            os.makedirs(Save_Dir)
            os.makedirs(Save_Dir + "/pos/")
            os.makedirs(Save_Dir + "/neg/")
    
        filename = os.path.basename(img_path)
        filename = filename +'_' +str(i)  + '.png'
        newPath = Save_Dir+'/neg/' + filename
        cv.imwrite(newPath, resized_image)
#         shutil.copy(img_path, newPath)
        all_file_empi_anon_neg.append([empi_anon, newPath, race,marriage])
#         if i >= len(df_meta_scr_pos_rel)+1:
#             print("Neg loop broke out")
#             break

CoppFileToDirectory_Neg(save_dir)


# In[ ]:


len(all_file_empi_anon_neg)


# In[ ]:


neg_df = pd.DataFrame(all_file_empi_anon_neg, columns=['empi_anon', 'file_path','race','marriage'])
neg_df.to_csv('/home/jupyter-ihwan28/breast_simple_comparison_by_race/images/800x600/br12_456/neg_empi_path.csv')


# In[ ]:


list(df_meta_scr_neg_rel.columns)


# In[ ]:





# In[ ]:




