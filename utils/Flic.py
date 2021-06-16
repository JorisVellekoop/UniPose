import scipy.io as sio
import os
from PIL import Image
import matplotlib.pyplot as plt
import math
import numpy as np
import cv2
import utils.Mytransforms as Mytransforms

def guassian_kernel(size_w, size_h, center_x, center_y, sigma):
    gridy, gridx = np.mgrid[0:size_h, 0:size_w]
    D2 = (gridx - center_x) ** 2 + (gridy - center_y) ** 2
    return np.exp(-D2 / 2.0 / sigma / sigma)

class Flic():
    def __init__(self, root_dir, sigma, stride, transformer=None):
        self.width = 480
        self.heigth = 720
        
        self.sigma = sigma
        self.stride = stride
        
        self.root_dir = root_dir
        self.transformer = transformer
        self.sigma = sigma
        # mat2 = sio.loadmat(os.path.join(self.root_dir, 'validation.mat'))
        # mat2 = sio.loadmat("validation.mat")
        data = sio.loadmat(os.path.join(self.root_dir, 'joints.mat'))
        if 'training' in data:
            self.data_file = data['training'][0]
        elif 'testing' in data:
            self.data_file = data['testing'][0]

        self.keys = keys = ['lsho',
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
        
    def __getitem__(self, index):
        image_path= os.path.join(self.root_dir, self.data_file[index]['filepath'][0])
        image = cv2.imread(image_path)
        #image = np.array(Image.open(image_path))
        
        center = [368/2,368/2] ##THis i am unsure about

        all_coordinates = self.data_file[index]['coords']
        kpt = all_coordinates[np.isnan(all_coordinates) == False]
        kpt = kpt.reshape((2, int(len(kpt) / 2)))

        if image.shape[0] != 368 or image.shape[1] != 368:
            kpt[0, :] *= (368. / image.shape[1])
            kpt[1, :] *= (368. / image.shape[0])
            image = cv2.resize(image, (368, 368))

        height, width, _ = image.shape

        heatmap = np.zeros((int(height/self.stride), int(width/self.stride), int(kpt.shape[1]+1)), dtype=np.float32)
        for i in range(kpt.shape[1]):
            # resize from 368 to 46
            x = int(kpt[0, i]) * 1.0 / self.stride
            y = int(kpt[1, i]) * 1.0 / self.stride
            heat_map = guassian_kernel(size_h=int(height/self.stride),size_w=int(width/self.stride), center_x=x, center_y=y, sigma=self.sigma)
            heat_map[heat_map > 1] = 1
            heat_map[heat_map < 0.0099] = 0
            heatmap[:, :, i + 1] = heat_map
            
        heatmap[:, :, 0] = 1.0 - np.max(heatmap[:, :, 1:], axis=2)  # for background
        
        centermap = np.zeros((height, width, 1), dtype=np.float32)
        center_map = guassian_kernel(size_h=height, size_w=width, center_x=center[0], center_y=center[1], sigma=3)
        center_map[center_map > 1] = 1
        center_map[center_map < 0.0099] = 0
        centermap[:, :, 0] = center_map

        image = Mytransforms.normalize(Mytransforms.to_tensor(image), [0, 0, 0],
                                     [265, 265, 265])
        heatmap   = Mytransforms.to_tensor(heatmap)
        centermap = Mytransforms.to_tensor(centermap)

        return image, heatmap, centermap, image_path, 0, 0

    def __len__(self):
        return len(self.data_file) #TODO

