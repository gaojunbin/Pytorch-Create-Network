"""
Created on 2020.6
Latest modify 2020.7
@Author: Junbin
@Note  : Data Pretreatment
"""
import torch
import numpy as np
import cv2

class NumpyToTensor(object):
    r"""
    transform the data from numpy to tensor
    """
    def __call__(self, data):
        return torch.from_numpy(data).float()

class CVReshape(object):
    r"""
    reshape the image read by cv2
    """
    def __call__(self, data:np.ndarray) -> np.ndarray:
        # here if use cv2 to get the data, we must reshape the size of data array to [channel,width,height]
        return data.reshape(3,32,32)

class CVReshape_inference(object):
    r"""
    reshape the image read by cv2
    """
    def __call__(self, data:np.ndarray) -> np.ndarray:
        # here if use cv2 to get the data, we must reshape the size of data array to [channel,width,height]
        return data.reshape(1,3,32,32)

class ResizePicture(object):
    r"""
    resize the picture to the wanted size(width,height)
    """
    def __call__(self,data:np.ndarray) -> np.ndarray:
        self.width = 32
        self.height = 32
        return cv2.resize(data,(self.width,self.height))