import scipy.io as sio
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
    def __init__(self, root_dir, sigma,stride, transform=None):
        self.width = 480
        self.heigth = 720
        
        self.sigma = sigma
        self.stride = stride
        
        self.root_dir = root_dir
        self.transform = transform
        self.sigma = sigma
        mat2 = sio.loadmat("test.mat")
        self.data_file = data = mat2['loading2'][0]
        
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
        
    def __getitem__(self,index):
        image_path= ("images/" + self.data_file[index]['filepath'][0])
        image = np.array(cv2.resize(cv2.imread(image_path),(368,368)))
        #image = np.array(Image.open(image_path))
        
        center = [368/2,368/2] ##THis i am unsure about
        
        height, width, _ = image.shape
        
        all_coordinates = self.data_file[index]['coords']
        real_coordinates = []
        for i in range(all_coordinates.shape[1]):
            if math.isnan(all_coordinates[0,i]) == False:
                #real_coordinates.append([self.keys[i],all_coordinates[0,i],all_coordinates[1,i]])
                real_coordinates.append([all_coordinates[0,i],all_coordinates[1,i]])
        kpt = real_coordinates
        
        heatmap = np.zeros((int(height/self.stride), int(width/self.stride), int(len(kpt)+1)), dtype=np.float32)
        for i in range(len(kpt)):
            # resize from 368 to 46
            x = int(kpt[i][0]) * 1.0 / self.stride
            y = int(kpt[i][1]) * 1.0 / self.stride
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
        
        image2 = image
        
        image = Mytransforms.normalize(Mytransforms.to_tensor(image), [0, 0, 0],
                                     [265, 265, 265])
        heatmap   = Mytransforms.to_tensor(heatmap)
        centermap = Mytransforms.to_tensor(centermap)
        
        image = image.permute(1,2,0)
        return image, heatmap, centermap, image_path, image2

flic = Flic("/images",1,1,None)

image, heatmap, centermap, image_path, image2 = flic.__getitem__(200)
plt.imshow(image.numpy())
plt.show()

