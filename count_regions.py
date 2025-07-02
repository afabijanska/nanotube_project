# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:46:34 2024

@author: an_fab
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.io import imread, imsave

path_results = './test/preds/' 
num_classes = 7 #7
frac = 42

# Define a dictionary mapping label values to RGB color codes

class_map = {
    1: 'wall',
    2: 'empty_core',
    3: 'amorphous',
    4: 'liquid',
    5: 'Sn_oxide',
    6: 'Sn_metallic_crystalline',
    7: 'Sn metallic defocused'
}

df = pd.DataFrame(columns = ['file'] + list(class_map.values()))

files = os.listdir(path_results)
#filtered_files = files
filtered_files = [file for file in files if file.startswith('_')]

for file in filtered_files:
    
    new_row = []
    case_id = file.strip('_').replace('.tif','')    
    print(case_id)
    new_row.append(case_id)        
    
    im = imread(path_results + file)
    im = im/frac
    
    total = np.sum(im > 0)
    
    for label in range(1, num_classes+1):
        num = np.sum(im == label)
        print(label, num/total*100)
        new_row.append(num/total*100)
        
    new_df = pd.DataFrame([new_row], columns=df.columns)
    df = pd.concat([df, new_df], ignore_index=True)
    
print(df)
    
df.to_csv(path_results + 'results.csv', sep=';', index=False)