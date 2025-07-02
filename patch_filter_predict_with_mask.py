# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:46:48 2024

@author: an_fab
"""

import os
import gc

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from skimage import io
from skimage.color import rgb2gray
from skimage.color import label2rgb

from helpers import write_hdf5

from keras import layers
from keras import regularizers

from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping


from keras.models import Model

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, LocallyConnected2D, LeakyReLU

from RandomBatchGenerator import RandomBatchGenerator

from tensorflow.keras import backend as K
#---------------------------------------------------------------------------

def patch_generator(patches, batch_size):
    num_patches = len(patches)
    while True:
        for offset in range(0, num_patches, batch_size):
            batch_patches = patches[offset:offset+batch_size]
            yield batch_patches

#---------------------------------------------------------------------------

def my_lab2rgb(img, labels, alpha=0.75):
    
    # Define a dictionary mapping label values to RGB color codes
    color_map = {
        1: [11, 97, 130],
        2: [15, 183, 222],
        3: [91, 213, 36],
        4: [225, 240, 29],
        5: [12, 28, 130],
        6: [130, 12, 14],
        7: [239, 138, 40],
        8: [112, 126, 255],
        9: [228, 16, 9],
        10: [175, 8, 149],
        11: [37, 248, 164],
        12: [226, 241, 29]
    }
    
    # Initialize the RGB output array with the grayscale image in all channels
    fin = np.stack([255*img, 255*img, 255*img], axis=-1).astype(np.float32)
    
    # Iterate over the color map and apply the color where labels match
    for label, color in color_map.items():
        mask = (labels == label)
        for i in range(3):  # Iterate over RGB channels
            fin[:, :, i][mask] = (1 - alpha) * fin[:, :, i][mask] + alpha * color[i]
    
    # Convert back to uint8 to represent an image
    return np.clip(fin, 0, 255).astype(np.uint8)


patch_size = 64

test_dir = './test' 

model_file = 'model_patch_filter.json'
best_weights_file = 'model_patch_filter.h5'


with open(model_file, 'r') as json_file:
    json_string = json_file.read()

model = model_from_json(json_string)
model.load_weights(best_weights_file)

files = os.listdir(test_dir + '/org/')
    
for file in files:
    
    print(file)
    img = io.imread(test_dir + '/org/' + file)
    
    if len(img.shape) == 2:
        img = img/255
    if len(img.shape) == 3:
        img = rgb2gray(img)
        
    img2 = np.copy(img)
    #img = img/255
    
    mask = io.imread(test_dir + '/tube_masks/' + file)
    
    #mask = rgb2gray(mask)
    
    if len(mask.shape) == 3:
        mask = rgb2gray(mask)
        
        
    img[mask==0] = 0
    
    (Y, X) = img.shape
    num = 0
    X_data = []
    
    for y in range(patch_size//2, Y - patch_size//2):
        
        y1 = y - patch_size//2
        y2 = y + patch_size//2
    
        for x in range(patch_size//2, X - patch_size//2):
    
            x1 = x - patch_size//2
            x2 = x + patch_size//2
        
            patch = img[y1:y2, x1:x2]
            X_data.append(patch)
    
            num = num + 1
    
    X_data = np.array(X_data, dtype = np.float16)
    X_data = np.reshape(X_data, (num, patch_size, patch_size, 1))

    # Create the generator
    batch_size = 4096  # Adjust based on your GPU memory capacity
    generator = patch_generator(X_data, batch_size)
    
    # Predict pixel labels using the generator
    steps = np.ceil(len(X_data) / batch_size)
    Y_pred = model.predict(generator, steps = steps)
    Y_pred = np.argmax(Y_pred, axis = -1)
    
    Y_fin = np.zeros((Y,X), dtype = np.uint8)
   
    num = 0
    
    for y in range(patch_size//2, Y - patch_size//2):
        
        for x in range(patch_size//2, X - patch_size//2):
            
            Y_fin[y, x] = Y_pred[num]
            num = num + 1
        
    plt.imshow(Y_fin)
    plt.show()
    
    #img_fin = label2rgb(Y_fin, image=img, bg_label=0, alpha=0.5)
    #img_fin = (img_fin * 255).astype(np.uint8)
    img_fin = my_lab2rgb(img2, Y_fin);
    plt.imshow(img_fin)
    plt.show()
    path = test_dir + '/preds/' + file
    path = path.replace('png','jpg')
    path = path.replace('tif','jpg')
    io.imsave(path, img_fin)
    
    path = test_dir + '/preds/' + '_' + file
    io.imsave(path, (Y_fin * 42).astype(np.uint8))
    
    del X_data, Y_pred, Y_fin, generator, img_fin, img, img2, mask
    gc.collect()
    K.clear_session()

       