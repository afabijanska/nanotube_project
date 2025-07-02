# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 15:46:48 2024

@author: an_fab
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from skimage import io
from skimage.color import rgb2gray
from skimage.morphology import disk

from skimage.filters.rank import maximum

from helpers import write_hdf5

from keras import layers
from keras import regularizers

from keras.utils import to_categorical
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from keras.models import Model

from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, LocallyConnected2D, LeakyReLU

from RandomBatchGenerator import RandomBatchGenerator

#-----------------------------------------------------------------------------

def residual_block(y, nb_channels, _strides=(1, 1), _project_shortcut=False):
    
    shortcut = y

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same', use_bias=False)(y)
    y = layers.LeakyReLU()(y)

    y = layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=False)(y)

    if _project_shortcut or _strides != (1, 1):
        shortcut = layers.Conv2D(nb_channels, kernel_size=(1, 1), strides=_strides, padding='same', use_bias=False)(shortcut)

    y = layers.add([shortcut, y])
    y = layers.LeakyReLU()(y)

    return y

#-----------------------------------------------------------------------------
# define residual model
    
def getSampleResidualModel(numClasses, shape):
    
    inputs = Input(shape=shape)
    
    conv1 = Conv2D(filters=64, kernel_size=(3,3), padding='same')(inputs)
    act1 = LeakyReLU(alpha=0.1)(conv1)
    pool1 = MaxPool2D(pool_size=(2,2))(act1)

    res1 = residual_block(pool1, 64) ;  
    pool2 = MaxPool2D(pool_size=(2,2))(res1)
    
    res2 = residual_block(pool2, 64)
    pool3 = MaxPool2D(pool_size=(2,2))(res2)
    
    conv4 = Conv2D(filters=64, kernel_size=(3,3), padding='same')(pool3)

    act4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPool2D(pool_size=(2,2))(act4)
    
    flat1 = Flatten()(pool4)
    #dens1 = Dense(256, activation='relu')(flat1)
    dens1 = Dropout(0.5)(Dense(256, activation='relu')(flat1))
    dens2 = Dense(numClasses, activation = 'softmax')(dens1)
    
    model = Model(inputs=inputs, outputs=dens2)
    model.compile(optimizer=Adam(lr = 0.0001, decay = 0),loss='categorical_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=Adam(lr = 0.01), loss=focal_loss(gamma=2., alpha=.50), metrics = ['accuracy'])
    model.summary()
    
    return model

#-----------------------------------------------------------------------------
#sample train 

patch_size = 64

train_dir = './train' 

model_file = 'model_patch_filter.json'
best_weights_file = 'model_patch_filter.h5'

X_data = []
Y_data = []

files = os.listdir(train_dir + '/org/')

num = 0
    
for file in files:
    
    print(file)
    img = io.imread(train_dir + '/org/' + file)

    if len(img.shape) == 2:
        img = img/255
    if len(img.shape) == 3:
        img = rgb2gray(img)

    print(img.shape)
    (Y, X) = img.shape
    
    label = io.imread(train_dir + '/labels2/' + file)
    
    if len(label.shape) == 3:
        label = rgb2gray(label)
    
    label = label//42    
    label =  maximum(label, disk(4))
    
    img[label==0]=0
    plt.imshow(img)
    plt.show()
    
    print(label.shape)
    print(np.unique(label))
    
    for y in range(patch_size//2 + 1, Y - patch_size//2, 20):
        
        y1 = y - patch_size//2
        y2 = y + patch_size//2
    
        for x in range(patch_size//2 + 1, X - patch_size//2, 20):
    
            x1 = x - patch_size//2
            x2 = x + patch_size//2
        
            patch = img[y1:y2, x1:x2]
            
            if patch.shape == (patch_size, patch_size):
                X_data.append(patch)
                Y_data.append(label[y,x])
                num = num + 1
        
X_data = np.array(X_data).astype(np.float16)
X_data = np.reshape(X_data, (num, patch_size, patch_size, 1))

Y_data = np.array(Y_data).astype(np.float16)
Y_data = to_categorical(Y_data)

write_hdf5(Y_data,'Y_train.h5')
write_hdf5(X_data, 'X_train.h5')

X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)

batch_size = 4*1024

generator_train = RandomBatchGenerator(X_train, Y_train, batch_size)
generator_val = RandomBatchGenerator(X_val, Y_val, batch_size//4)

#create callbacks for training
checkpointer = ModelCheckpoint(best_weights_file, 
                               verbose = 1, 
                               monitor = 'val_loss', 
                               mode = 'auto', 
                               save_best_only=True) #save at each epoch if the validation decreased

patienceCallBack = EarlyStopping(monitor = 'val_loss',
                                 patience = 30)

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

#change first parameter depending on number of classess you want to get
model = getSampleResidualModel(7, (patch_size, patch_size, 1))
    
json_string = model.to_json()
open(model_file, 'w').write(json_string)

history = model.fit(generator_train, 
                    validation_data = generator_val,
                    verbose = 1, 
                    callbacks = [checkpointer, patienceCallBack], 
                    epochs = 1000)

model.save_weights('last_patch_filter.h5')

# #-----------------------------------------------------------------------------
# #plot train history

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()