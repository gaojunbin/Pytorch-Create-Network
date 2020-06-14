import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import Data.DataPre as dp
import Data.DataReload as dr
import Network.Network as Network
import os
import argparse
import random
import yaml

parser = argparse.ArgumentParser(description='params about training')
parser.add_argument('--config', default='Config/Config.yaml', type=str)

def train(args):
    
    train_transforms = transforms.Compose([
        dp.NumpyToTensor()
    ])
    test_transforms = transforms.Compose([
        dp.NumpyToTensor()
    ])

    mydataset_train = dr.MyDataset(num_class = args.num_class, is_train = True, root_path=args.Dataset_root_path, transform=train_transforms)
    train_loader = Data.DataLoader(
        dataset=mydataset_train,      # 数据集
        batch_size=args.BATCH_SIZE,   # mini batch size
        shuffle=True,                 # 是否打乱数据集
        num_workers=args.num_workers,        
    )

    mydataset_test = dr.MyDataset(num_class = args.num_class, is_train = False, root_path=args.Dataset_root_path, transform=train_transforms)
    test_loader = Data.DataLoader(
        dataset=mydataset_test,      # 数据集
        batch_size=args.BATCH_SIZE, # mini batch size
        shuffle=False,              # 是否打乱数据集
        num_workers=args.num_workers,        
    )

    network = Network.Network(select_net = args.select_net, use_gpu = args.use_gpu)
    net = network.get_net()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate) 
    loss_func = nn.CrossEntropyLoss()                   
    for epoch in range(args.EPOCH):
        for step,(b_x,b_y) in enumerate(train_loader):
            network.train_or_test(is_train = True)
            output = net(b_x)
            loss = loss_func(output, b_y)   # cross entropy loss
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients
            if step % 50 == 0:
                network.train_or_test(is_train = False)
                # test_output = net(test_loader)
                # pred_y = torch.max(test_output, 1)[1].data.numpy()
                # accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                # print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
def main():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    print("\n**************************")
    for k, v in config['params'].items():
        setattr(args, k, v)
        print('\n[%s]:'%(k), v)
    print("\n**************************\n")

    #create checkpoint folder to save model
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    train(args)

if __name__ == "__main__":
    main()