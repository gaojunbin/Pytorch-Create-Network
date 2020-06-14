import torch
import numpy as np

class NumpyToTensor(object):
    def __call__(self, data):
        return torch.from_numpy(data).float()