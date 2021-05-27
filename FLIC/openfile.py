# -*- coding: utf-8 -*-
"""
Created on Mon May 10 11:53:48 2021

@author: remco
"""
import numpy as np
import scipy.io as sio
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import math

nan = float('nan')

keys_dict = {'lsho':1,
'lelb':2,
'lwri':3,
'rsho':4,
'relb':5,
'rwri':6,
'lhip':7,
'lkne':8,
'lank':9,
'rhip':10,
'rkne':11,
'rank':12,
'leye':13,
'reye':14,
'lear':15,
'rear':16,
'nose':17,
'msho':18,
'mhip':19,
'mear':20,
'mtorso':21,
'mluarm':22,
'mruarm':23,
'mllarm':24,
'mrlarm':25,
'mluleg':26,
'kmruleg':27,
'mllleg':28,
'mrlleg':29}

keys = ['lsho',
'lelb',
'lwri',
'rsho',
'relb',
'rwri',
'lhip',
'lkne',
'lank',
'rhip',
'rkne',
'rank',
'leye',
'reye',
'lear',
'rear',
'nose',
'msho',
'mhip',
'mear',
'mtorso',
'mluarm',
'mruarm',
'mllarm',
'mrlarm',
'mluleg',
'kmruleg',
'mllleg',
'mrlleg']

mat2 = sio.loadmat("test.mat")
data = mat2['loading2'][0]

image_number = 4


filepath = ('images/'+data[image_number]['filepath'][0])
all_coordinates = data[image_number]['coords']



#show image
im = Image.open(filepath)
plt.imshow(im)

# print('poselet = ',data[image_number]['poselet_hit_idx'])
# print('coords = ',all_coordinates)
# print('filepath = ',filepath)
# print('istrain = ',data[image_number]['istrain'])
# print('istest = ',data[image_number]['istest'])
# print('torsobox = ',data[image_number]['torsobox'])


real_coordinates = []
for i in range(all_coordinates.shape[1]):
    if math.isnan(all_coordinates[0,i]) == False:
        real_coordinates.append([keys[i],all_coordinates[0,i],all_coordinates[1,i]])

for j in range(len(real_coordinates)):
    plt.plot(real_coordinates[j][1],real_coordinates[j][2],'rx')
    
