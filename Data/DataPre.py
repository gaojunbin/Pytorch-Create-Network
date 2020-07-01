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
        return data.reshape(1,-1,224,224)