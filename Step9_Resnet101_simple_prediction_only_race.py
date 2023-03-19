#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="6" # Choose which GPUs by checking current use with nvidia-smi
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.applications import ResNet101V2
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import metrics
import numpy as np
import pandas as pd
from keras import backend as K
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
print("*************************************")
print(gpus)
print("*************************************")

def preprocess(images, labels):
    return tf.keras.applications.resnet_v2.preprocess_input(images), labels

### This code is for the testing phase. Only testdir folder needs to be used. traindir and valdir are not used in this step. 
print("Black Population AUC measure started")
traindir = './train_800_600_simple_comparison_birad'
valdir = './val_800_600_simple_comparison_birad'
testdir = './test_800_600_simple_comparison_Black'
dirName = '800_600'



buffersize = 3
#im_dim = 512
im_dim_x = 800
im_dim_y = 600
cutoff=0.5
train = tf.keras.preprocessing.image_dataset_from_directory(
    traindir, image_size=(im_dim_x, im_dim_y), batch_size=10)
val = tf.keras.preprocessing.image_dataset_from_directory(
    valdir, image_size=(im_dim_x, im_dim_y), batch_size=10)
test = tf.keras.preprocessing.image_dataset_from_directory(
    testdir, image_size=(im_dim_x, im_dim_y), batch_size=20)

test_ds = test.map(preprocess)
train_ds = train.map(preprocess)
val_ds = val.map(preprocess)
train_ds = train_ds.prefetch(buffer_size=buffersize)
val_ds = val_ds.prefetch(buffer_size=buffersize)



## do not follow this code, They will not be used.
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

eachParameter = []
PerformanceRecord = []
for i in range(1):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        metr = [metrics.BinaryAccuracy(name='accuracy', threshold=cutoff), metrics.AUC(name='auc'), metrics.Precision(name='precision'), metrics.Recall(name='recall')]
        ## load model generated in the previous training step. This code is only for testing purpose.
        model = tf.keras.models.load_model('saved_model_resnet101_simple800_600/resnet152v2_1')
        model.compile(metrics=metr)
        testloss, testaccuracy, testauc, precision, recall = model.evaluate(test_ds)

        F1 = 2*float(precision)*float(recall)/(float(precision) + float(recall))
        print('Test accuracy :', testaccuracy)
        print('Test AUC :', testauc)
        print('Test F1 :', F1)
        print('Test precision :', precision)
        print('Test recall :', recall)
        PerformanceRecord.append([testaccuracy, testauc,F1, precision, recall ])
    
## Prediction probability and predicted labels are extracted for analysis.
predicted_probs = np.array([])
true_classes =  np.array([])
IterationChecker = 0
for images, labels in test_ds:
    if IterationChecker == 0:
        predicted_probs = model(images)
        true_classes = labels.numpy()

    IterationChecker += 1

    predicted_probs = np.concatenate([predicted_probs,
                       model(images)])
    true_classes = np.concatenate([true_classes, labels.numpy()])
# Since they are sigmoid outputs, you need to transform them into classes with a threshold, i.e 0.5 here:
predicted_classes = [1 * (x[0]>=cutoff) for x in predicted_probs]
# confusion matrix etc:
conf_matrix = tf.math.confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)
predicted_probs=np.squeeze(predicted_probs)
predicted_classes = np.array(predicted_classes)
true_classes=np.squeeze(true_classes)
##Extracted information is stored into csv files so that they will be visualized.
summedResults = np.stack((predicted_probs,predicted_classes,true_classes), axis = 1)
np.savetxt("Resnet101_simple_comp_EMBED_Black.csv", summedResults, delimiter=',', header="predicted_probabilty,predicted_classes,true_classes", comments="")
# In[ ]:


### Testing over White race is performed. All the steps are exactly same as before. 
print("White Population AUC measure started")
traindir = './train_800_600_simple_comparison_birad'
valdir = './val_800_600_simple_comparison_birad'
testdir = './test_800_600_simple_comparison_White'
dirName = '800_600'



buffersize = 3
#im_dim = 512
im_dim_x = 800
im_dim_y = 600
cutoff=0.5
train = tf.keras.preprocessing.image_dataset_from_directory(
    traindir, image_size=(im_dim_x, im_dim_y), batch_size=10)
val = tf.keras.preprocessing.image_dataset_from_directory(
    valdir, image_size=(im_dim_x, im_dim_y), batch_size=10)
test = tf.keras.preprocessing.image_dataset_from_directory(
    testdir, image_size=(im_dim_x, im_dim_y), batch_size=20)

test_ds = test.map(preprocess)
train_ds = train.map(preprocess)
val_ds = val.map(preprocess)
train_ds = train_ds.prefetch(buffer_size=buffersize)
val_ds = val_ds.prefetch(buffer_size=buffersize)




def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
#im_dim = 512

# for i in np.arange(0, 4.5, 0.5):
# ...     print(i, end=', ')
eachParameter = []
PerformanceRecord = []
for i in range(1):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        metr = [metrics.BinaryAccuracy(name='accuracy', threshold=cutoff), metrics.AUC(name='auc'), metrics.Precision(name='precision'), metrics.Recall(name='recall')]
        model = tf.keras.models.load_model('saved_model_resnet101_simple800_600/resnet152v2_1')
        model.compile(metrics=metr)
        testloss, testaccuracy, testauc, precision, recall = model.evaluate(test_ds)
    # predictions = model.predict(test_ds)
    # y_pred = (predictions > 0.5)

        F1 = 2*float(precision)*float(recall)/(float(precision) + float(recall))
        print('Test accuracy :', testaccuracy)
        print('Test AUC :', testauc)
        print('Test F1 :', F1)
        print('Test precision :', precision)
        print('Test recall :', recall)
        PerformanceRecord.append([testaccuracy, testauc,F1, precision, recall ])
    

predicted_probs = np.array([])
true_classes =  np.array([])
IterationChecker = 0
for images, labels in test_ds:
    if IterationChecker == 0:
        predicted_probs = model(images)
        true_classes = labels.numpy()

    IterationChecker += 1

    predicted_probs = np.concatenate([predicted_probs,
                       model(images)])
    true_classes = np.concatenate([true_classes, labels.numpy()])
# Since they are sigmoid outputs, you need to transform them into classes with a threshold, i.e 0.5 here:
predicted_classes = [1 * (x[0]>=cutoff) for x in predicted_probs]
# confusion matrix etc:
conf_matrix = tf.math.confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)
predicted_probs=np.squeeze(predicted_probs)
predicted_classes = np.array(predicted_classes)
true_classes=np.squeeze(true_classes)
summedResults = np.stack((predicted_probs,predicted_classes,true_classes), axis = 1)
np.savetxt("Resnet101_simple_comp_EMBED_White.csv", summedResults, delimiter=',', header="predicted_probabilty,predicted_classes,true_classes", comments="")

### Testing over ASIAN race images are performed.
print("Asian Population AUC measure started")
traindir = './train_800_600_simple_comparison_birad'
valdir = './val_800_600_simple_comparison_birad'
testdir = './test_800_600_simple_comparison_Asian'
dirName = '800_600'



buffersize = 3
#im_dim = 512
im_dim_x = 800
im_dim_y = 600
cutoff=0.5
train = tf.keras.preprocessing.image_dataset_from_directory(
    traindir, image_size=(im_dim_x, im_dim_y), batch_size=10)
val = tf.keras.preprocessing.image_dataset_from_directory(
    valdir, image_size=(im_dim_x, im_dim_y), batch_size=10)
test = tf.keras.preprocessing.image_dataset_from_directory(
    testdir, image_size=(im_dim_x, im_dim_y), batch_size=20)

test_ds = test.map(preprocess)
train_ds = train.map(preprocess)
val_ds = val.map(preprocess)
train_ds = train_ds.prefetch(buffer_size=buffersize)
val_ds = val_ds.prefetch(buffer_size=buffersize)




def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
#im_dim = 512

# for i in np.arange(0, 4.5, 0.5):
# ...     print(i, end=', ')
eachParameter = []
PerformanceRecord = []
for i in range(1):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        metr = [metrics.BinaryAccuracy(name='accuracy', threshold=cutoff), metrics.AUC(name='auc'), metrics.Precision(name='precision'), metrics.Recall(name='recall')]
        model = tf.keras.models.load_model('saved_model_resnet101_simple800_600/resnet152v2_1')
        model.compile(metrics=metr)
        testloss, testaccuracy, testauc, precision, recall = model.evaluate(test_ds)
    # predictions = model.predict(test_ds)
    # y_pred = (predictions > 0.5)

        F1 = 2*float(precision)*float(recall)/(float(precision) + float(recall))
        print('Test accuracy :', testaccuracy)
        print('Test AUC :', testauc)
        print('Test F1 :', F1)
        print('Test precision :', precision)
        print('Test recall :', recall)
        PerformanceRecord.append([testaccuracy, testauc,F1, precision, recall ])
    

predicted_probs = np.array([])
true_classes =  np.array([])
IterationChecker = 0
for images, labels in test_ds:
    if IterationChecker == 0:
        predicted_probs = model(images)
        true_classes = labels.numpy()

    IterationChecker += 1

    predicted_probs = np.concatenate([predicted_probs,
                       model(images)])
    true_classes = np.concatenate([true_classes, labels.numpy()])
# Since they are sigmoid outputs, you need to transform them into classes with a threshold, i.e 0.5 here:
predicted_classes = [1 * (x[0]>=cutoff) for x in predicted_probs]
# confusion matrix etc:
conf_matrix = tf.math.confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)
predicted_probs=np.squeeze(predicted_probs)
predicted_classes = np.array(predicted_classes)
true_classes=np.squeeze(true_classes)
summedResults = np.stack((predicted_probs,predicted_classes,true_classes), axis = 1)
np.savetxt("Resnet101_simple_comp_EMBED_Asian.csv", summedResults, delimiter=',', header="predicted_probabilty,predicted_classes,true_classes", comments="")


## Testing over unclassified races are performed.
print("Unknown Population AUC measure started")
traindir = './train_800_600_simple_comparison_birad'
valdir = './val_800_600_simple_comparison_birad'
testdir = './test_800_600_simple_comparison_Unknown'
dirName = '800_600'



buffersize = 3
#im_dim = 512
im_dim_x = 800
im_dim_y = 600
cutoff=0.5
train = tf.keras.preprocessing.image_dataset_from_directory(
    traindir, image_size=(im_dim_x, im_dim_y), batch_size=10)
val = tf.keras.preprocessing.image_dataset_from_directory(
    valdir, image_size=(im_dim_x, im_dim_y), batch_size=10)
test = tf.keras.preprocessing.image_dataset_from_directory(
    testdir, image_size=(im_dim_x, im_dim_y), batch_size=20)

test_ds = test.map(preprocess)
train_ds = train.map(preprocess)
val_ds = val.map(preprocess)
train_ds = train_ds.prefetch(buffer_size=buffersize)
val_ds = val_ds.prefetch(buffer_size=buffersize)




def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
#im_dim = 512

# for i in np.arange(0, 4.5, 0.5):
# ...     print(i, end=', ')
eachParameter = []
PerformanceRecord = []
for i in range(1):
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        metr = [metrics.BinaryAccuracy(name='accuracy', threshold=cutoff), metrics.AUC(name='auc'), metrics.Precision(name='precision'), metrics.Recall(name='recall')]
        model = tf.keras.models.load_model('saved_model_resnet101_simple800_600/resnet152v2_1')
        model.compile(metrics=metr)
        testloss, testaccuracy, testauc, precision, recall = model.evaluate(test_ds)
    # predictions = model.predict(test_ds)
    # y_pred = (predictions > 0.5)

        F1 = 2*float(precision)*float(recall)/(float(precision) + float(recall))
        print('Test accuracy :', testaccuracy)
        print('Test AUC :', testauc)
        print('Test F1 :', F1)
        print('Test precision :', precision)
        print('Test recall :', recall)
        PerformanceRecord.append([testaccuracy, testauc,F1, precision, recall ])
    

predicted_probs = np.array([])
true_classes =  np.array([])
IterationChecker = 0
for images, labels in test_ds:
    if IterationChecker == 0:
        predicted_probs = model(images)
        true_classes = labels.numpy()

    IterationChecker += 1

    predicted_probs = np.concatenate([predicted_probs,
                       model(images)])
    true_classes = np.concatenate([true_classes, labels.numpy()])
# Since they are sigmoid outputs, you need to transform them into classes with a threshold, i.e 0.5 here:
predicted_classes = [1 * (x[0]>=cutoff) for x in predicted_probs]
# confusion matrix etc:
conf_matrix = tf.math.confusion_matrix(true_classes, predicted_classes)
print(conf_matrix)
predicted_probs=np.squeeze(predicted_probs)
predicted_classes = np.array(predicted_classes)
true_classes=np.squeeze(true_classes)
summedResults = np.stack((predicted_probs,predicted_classes,true_classes), axis = 1)
np.savetxt("Resnet101_simple_comp_EMBED_Unknown.csv", summedResults, delimiter=',', header="predicted_probabilty,predicted_classes,true_classes", comments="")



