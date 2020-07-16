# -*- coding: utf-8 -*-
"""
Created on 2020.6
Latest modify 2020.7
@Author: Junbin
@Note  : Inference
"""
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import Data.DataPre as dp
import Network.Network as Network
import cv2
import os
import argparse
import random
import yaml

parser = argparse.ArgumentParser(description='params about inference')
parser.add_argument('--config', default='Config/Config.yaml', type=str)

def inference(inference_img:np.ndarray,args) -> int:
    # inference data Pretreatment factory
    inference_transforms = transforms.Compose([
        dp.ResizePicture(),
        dp.CVReshape(),
        dp.NumpyToTensor()
    ])
    inference_tensor = inference_transforms(inference_img)

    # initial network
    network = Network.Network(select_net = args.select_net)
    net = network.get_net()

    # restore the parameters 
    try:
        net.load_state_dict(torch.load(args.checkpoint_path+'/'+args.checkpoint_inf))
    except:
        print("**no model can be found**\n")

    # inference
    network.train_or_test(is_train = False)
    inference_output = net(inference_tensor)
    inference_result = torch.max(inference_output, 1)[1].data.numpy()[0]
    return inference_result

def main():
    # get the config parameters and print
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    print("\n**************************")
    for k, v in config['inference'].items():
        setattr(args, k, v)
        print('\n[%s]:'%(k), v)
    print("\n**************************\n")

    inference_img = cv2.imread('./Data/Dataset/dog/copper0_95.png')
    result = inference(inference_img,args)
    print(result)

if __name__ == "__main__":
    main()