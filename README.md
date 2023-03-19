
## Author

InChan Hwang(ihwang@students.kennesaw.edu)

# Impact of Multi-source Data Augmentation on Performance of Convolutional Neural Networks for Abnormality Classification in Mammography

CBIS-DDSM dataset is widespread in the deep learning community especially for those who develop deep learning models classifying and detecting breast cancer tumors. What happens when those old film based CBIS-DDSM/DDSM and full digital mammograms are mixed and trained to avoid data scarcity issue?  

## Description

Employment of the deep learning models to detect breast cancer tumors in the mammograms are actively being researched across the world. It increases patients survival rates, it helps decide early and accurate treatment on breast cancer. In order to train deep learning models, a massive amount of breast cancer models is required for higher performance of the model. However, data scarcity issue in breast cancer mammograms is also common as well as other cancer types. Although the most recent release of EMBED dataset, Full Digital Mammogram, resolves such issue a little, we need to investigate the consequence when full digital mammograms are mixed with film based mammograms, such as CBIS-DDSM/DDSM to increase the training volume. It is the 1st study to examine the effect of the mixture of EMBED and CBIS-DDSM/DDSM.
## Getting Started

### Dependencies

This code has been tested under Ubuntu 20.04 with Tensorflow. Go to the website for its installation details (https://www.tensorflow.org/install/pip)\
Required packages for preprocessing mammograms : Pandas, Numpy, scikit-learn, pydicom, scikit-image, pillow

LJPEG codec and its original code has come from : https://github.com/aaalgo/ljpeg


### Installing
To install tensorflow, go to the webpage : https://www.tensorflow.org/install/pip. 
To install Python packages, run pip command like below.

pip install pandas numpy scikit-learn pydicom scikit-image pillow pylibjpeg pylibjpeg-libjpeg.

Visit the github page to install LJPEG codec : https://github.com/aaalgo/ljpeg 
 
### File Descriptions
1st_stage_convert_dicom_png.ipynb : This file contains step-by-step Jupyter notebook style python code blocks with comments.
It reads clinical data and mammogram metadata, showing you how to properly filter clinical information, and join them with mammogram metadata file.
It give you an example how to locate mammograms accordingt to their BIRAD number.
It also shows you an example how to read DICOM format images and convert them into 16 BIT color PNGs. It also rescales different mammograms' resolutions into uniform single mammogram resolution. This file also produces metadata files to trace preprocessed mammograms.

2nd_stage_split_dataset.ipynb : This file splits dataset into Train, Validation, Test sets. This example made sure that patient ID colums are stratified across train, validation, and test sets. This file needs output files from the previous stages. To ensure the balance the dataset between positive and negative cases, shuffle() function was used to select benign cases randomly to ensure 1:1 ratio for positive and negative cases. The hyperparameter "random_state" affects the model performance. "random_state=5" was chosen for a reasonable performance of models. However, a better random_state value may exist. Mix ratio is 60:20:20.

3rd_stage_modelling.ipynb : This file contains Tensorflow code that builds RESNET50V2 based transfer learning model code. This file reads train, validation, test folders and trains a customized RESNET50V2 model, store the trained model into a file, also records performance metrics into CSV files. 


3.5th_stage_ljpeg_to_png.zip : This file contains LJPEG codec and the python script that converts DDSM LJPEG images into 16BIT PNGs. Although I compiled LJPEG codec, you may need to recompile it for your own system. Visit and ask the author how to compile it : https://github.com/aaalgo/ljpeg. If my binary works, it is fine. 

massconvert.py script will use a binary executable in 'ljpeg' folder and read DDSM LJPEG images in 'normal' folder and convert them into 16Bit PNGs in 'negative' folder. 

4th_stage_CBIS_DDSM_data_prep.ipynb : It preprocesses a public dataset, CBIS-DDSM/DDSM, to mix it with the most recently released EMBED dataset. It reads metadata.csv of CBIS-DDSM and labels each mammogram with benign and malignant. It also converts DICOM files into 16 Bit PNGs. 

5th_stage_CBIS_DDSM_EMBED.ipynb : It mixes EMBED and CBIS-DDSM/DDSM together and produce mixed train, mixed validation sets. However, test set only contain EMBED dataset only. Mix ratio is 60:20:20.

6th_stage_modelling_combined.ipynb : Resnet models are trained by mixed train sets, validation sets, and test sets. store the trained model into a file, also records performance metrics into CSV files.

7th_Stage_Simple_By_Race.ipynb : this step may look like 1st stage but adding racial information to mammography images.

8th_Stage_Racial_Test_Split_Folder.ipynb : This step also looks pretty similar to 2nd stage but split mammogram images by different races.

9th_Stage_Resnet101_simple_prediction_only_race.py : This one looks pretty similar to 3rd stage but only testing over testsets in regards to different races.

10th_stage_AUC_BY_RACE.ipynb : This file draws ROC AUC curves in regards to different races.

Your computer system is different from mine. You will definitely run into problems reproducing the outputs. Do not hesitate to contact me if you are having trouble running the code.



## Help

Contact : InChan Hwang(ihwang@students.kennesaw.edu)


## Version History

* 0.1
    * Initial Release

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments
HITI Lab at Emory University (https://hitilab.com/)
