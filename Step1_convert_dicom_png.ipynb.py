#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[17]:



### install packages like numpy, pandas, pillow, matplotlib, pydicom, skimage, scikit-learn, 
##pip install pylibjpeg pylibjpeg-libjpeg pydicom
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from sklearn.utils import shuffle
import shutil
import pydicom
from pydicom import dcmread
from skimage.transform import resize
from PIL import Image


# In[18]:





# In[ ]:





# In[ ]:





# In[19]:


## Read clinical and metadata rows
magview_path = '/media/careinfolab/CI_Lab/tables/EMBED_OpenData_clinical.csv'
metadata_path = '/media/careinfolab/CI_Lab/tables/EMBED_OpenData_metadata.csv'
df_mag = pd.read_csv(magview_path)


# In[20]:


## Read clinical data and convert date columns into an appropriate style.
df_mag = pd.read_csv(magview_path)
df_mag['study_date_anon'] = pd.to_datetime(df_mag['study_date_anon'], errors='coerce', format= '%Y-%m-%d')
list(df_mag.columns)


# In[21]:


## read file metadata files 
df_meta = pd.read_csv(metadata_path)
list(df_meta.columns)


# In[22]:


df_meta.anon_dicom_path[0]


# In[ ]:





# In[23]:


## Public EMBED dataset has only cohort 1 and 2 

print(df_meta.cohort_num.unique())
print(df_mag.cohort_num.unique())


# In[24]:


# Get diagnostic exams with BIRADS 4/5/6
df_mag_diag_pos = df_mag[df_mag.asses.isin(['S','M','K']) & 
                         df_mag.desc.str.contains('diag', case=False)]

# Get and rename relevant columns to prepare for merge with screening exams
df_mag_diag_pos_empi = df_mag_diag_pos[['empi_anon', 
                                        'acc_anon', 
                                        'numfind', 
                                        'bside', 
                                        'study_date_anon', 
                                        'asses']]

df_mag_diag_pos_empi.columns = ['empi_anon', 
                                'acc_anon_diag', 
                                'diag_num', 
                                'diag_side', 
                                'diag_study_date', 
                                'diag_asses']

df_mag_diag_pos_empi


# In[25]:


# Get screening exams and left merge on empi_anon(Patient_ID) with df_mag_diag_pos_empi
df_mag_scr = df_mag[df_mag.desc.str.contains('screen', case=False)]
df_mag_scr_pos = df_mag_diag_pos_empi.merge(df_mag_scr, on='empi_anon', how='left')

df_mag_scr_pos


# In[26]:


# Keep only screening exams with time diff less than +180 days between diagnosis date and the most recent exam day.
df_mag_scr_pos = df_mag_scr_pos.loc[(df_mag_scr_pos.side == df_mag_scr_pos.diag_side)]
df_mag_scr_pos['study_date_diff'] = df_mag_scr_pos.diag_study_date - df_mag_scr_pos.study_date_anon

df_mag_scr_pos_rel = df_mag_scr_pos.loc[(df_mag_scr_pos.study_date_diff.dt.days >= 0) & 
                                        (df_mag_scr_pos.study_date_diff.dt.days <= 180)]



df_mag_scr_pos_rel


# In[27]:


# Get relevant columns from df_meta or merge entire dataset
df_meta_rel = df_meta[['empi_anon', 
                       'acc_anon', 
                       'ViewPosition', 
                       'ImageLateralityFinal', 
                       'FinalImageType', 
                       'png_path', 
                       'num_roi',
                       'anon_dicom_path',
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
print(len(df_meta_scr_pos_rel))


# In[28]:



##fixing dicom file paths to the local ones.

def anon_dicom_path_fix(DICOMPathStr):
    return DICOMPathStr.replace('/mnt/NAS2/mammo/anon_dicom', '/media/careinfolab/CI_Lab')

df_meta_scr_pos_rel['anon_dicom_path_local']=df_meta_scr_pos_rel['anon_dicom_path'].apply(anon_dicom_path_fix)


# In[29]:



## See whether it displays local paths for dicom files    
df_meta_scr_pos_rel['anon_dicom_path_local'][0]


# In[ ]:


len(df_meta_scr_pos_rel) ## Display the number of rows it has


# In[30]:



def ConvertFromDCMtoPNG(srcPath, dstFolderPath):
    im_dim_x = 800
    im_dim_y = 600

    im = pydicom.dcmread(srcPath)
    print(srcPath + "'s resolution is " + str(im.pixel_array.shape))
    im = im.pixel_array.astype(float)
    ## anti aliasing is true. It is resized into 800 * 600. 
    im = resize(im, (im_dim_x, im_dim_y), anti_aliasing=True)

    ## Rescaled into gray scale image. 
    rescaled_image = (np.maximum(im, 0)/im.max())*65536

    final_image= np.uint16(rescaled_image)
    # Mammograms are converted into 16bit gray scale images    
    print(srcPath + "'s resolution rescaled to " + str(final_image.shape))


    final_image = Image.fromarray(final_image)
    final_image.save(dstFolderPath)


# In[31]:



## Mammograms are read from DICOM, resized and color depths are rescaled and stored into seperate folders.
df = df_meta_scr_pos_rel
df = shuffle(df)
all_rois_list = []
all_file_empi_anon_pos = []
save_dir = '/home/careinfolab/FIR_Inchan/breast_simple_comparison/images/800x600/br12_456/'

def CoppFileToDirectory_Pos(Save_Dir):
    for i in range(len(df)):
        if not os.path.exists(Save_Dir):
        # if the demo_folder directory is not present
        # then create it.
            os.makedirs(Save_Dir)
            os.makedirs(Save_Dir + "/pos/")
            os.makedirs(Save_Dir + "/neg/")
        
        img_path = df.anon_dicom_path_local[i]
        empi_anon = df.empi_anon[i]
        filename = os.path.basename(img_path)
        filename = filename +'_' +str(i) + '.png'
        newPath = Save_Dir+'/pos/' + filename
        ConvertFromDCMtoPNG(img_path,newPath)
    
        
        all_file_empi_anon_pos.append([empi_anon, newPath])

CoppFileToDirectory_Pos(save_dir)


# In[32]:


## Create a seperate metadata file for converted malignant mammogram images
pos_df = pd.DataFrame(all_file_empi_anon_pos, columns=['empi_anon', 'file_path'])
pos_df.to_csv('/home/careinfolab/FIR_Inchan/breast_simple_comparison/images/800x600/br12_456/pos_empi_path.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[33]:


# Get screening exams with BIRADS 1/2
df_mag_scr_neg = df_mag[df_mag.desc.str.contains('screen', case=False) & 
                        df_mag.asses.isin(['N','B'])]

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

df_meta_scr_neg_rel


# In[34]:


df_meta_scr_neg_rel['anon_dicom_path_local']=df_meta_scr_neg_rel['anon_dicom_path'].apply(anon_dicom_path_fix)


# In[35]:


## See whether it displays local paths for dicom files    
df_meta_scr_neg_rel['anon_dicom_path_local'][0]


# In[36]:


len(df_meta_scr_neg_rel) # see the number of rows it has


# In[37]:


## Read DICOM files and store them into 16Bit pngs with 800 x 600 resolutions.
df = df_meta_scr_neg_rel
df = shuffle(df)
all_rois_list = []
all_file_empi_anon_neg = []
save_dir = '/home/careinfolab/FIR_Inchan/breast_simple_comparison/images/800x600/br12_456/'

def CoppFileToDirectory_Neg(Save_Dir):
    for i in range(len(df)):
        if not os.path.exists(Save_Dir):
        # if the demo_folder directory is not present
        # then create it.
            os.makedirs(Save_Dir)
            os.makedirs(Save_Dir + "/pos/")
            os.makedirs(Save_Dir + "/neg/")
        
        img_path = df.anon_dicom_path_local[i]
        empi_anon = df.empi_anon[i]
        filename = os.path.basename(img_path)
        filename = filename +'_' +str(i) + '.png'
        newPath = Save_Dir+'/neg/' + filename
        ConvertFromDCMtoPNG(img_path,newPath)
    
        
        all_file_empi_anon_neg.append([empi_anon, newPath])

CoppFileToDirectory_Neg(save_dir)


# In[39]:


# create a metadata file for benign mammograms.
neg_df = pd.DataFrame(all_file_empi_anon_neg, columns=['empi_anon', 'file_path'])
neg_df.to_csv('/home/careinfolab/FIR_Inchan/breast_simple_comparison/images/800x600/br12_456/neg_empi_path.csv')


# In[ ]:




