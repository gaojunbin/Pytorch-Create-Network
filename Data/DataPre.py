import torch
import numpy as np
import cv2

class NumpyToTensor(object):
    def __call__(self, data):
        return torch.from_numpy(data).float()

class CVReshape(object):
    r'''
    reshape the image read by cv2
    '''
    def __call__(self, data:np.ndarray) -> np.ndarray:
        return data.reshape(1,-1,224,224)