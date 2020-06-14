# -*- coding: utf-8 -*-
"""
Created on 2020.6
latest modify 2020.6
@author: Junbin
@note: Dataset Reload
"""

import torch
import torch.utils.data as Data
import numpy as np
import os
import cv2
import random

class MyDataset(Data.Dataset):
    r"""my dataset, derived from torch.utils.data.DataSet\
    file:
        DataReload.py
        Datapre.py
        Dataset
            class1
                img1
                img2
                ...
            class2
            ...
    """
    def __init__(self, num_class, is_train, root_path='./Dataset', transform=None):
        #if transform is given, we transoform data using
        assert os.path.exists(root_path),'Dataset is not exists or can not be found:'+root_path
        assert len(os.listdir(root_path))==num_class,'number of class in dataset is wrong'
        class_name = os.listdir(root_path)
        class_index = [_ for _ in range(num_class)]
        self.class_dict = dict(zip(class_index,class_name))
        self.num_class = num_class
        self.transform = transform
        self.is_train = is_train
        self.data = []
        self.lable = []
        for each_class in self.class_dict:
            ls_dir = root_path + '/' + str(self.class_dict[each_class]) + '/'
            dirs = os.listdir(ls_dir)
            num = len(dirs)
            train_num = int(0.7 * num)
            test_num = num - train_num
            train_index = sorted(random.sample([_ for _ in range(num)],train_num))
            for i in range(num):
                img = cv2.imread(ls_dir+dirs[i]).reshape(-1,224,224)
                if img is None :
                    print("%s图片不存在，已跳过"%(ls_dir + dirs[i]))
                    continue
                else:
                    if self.is_train == True:
                        if (int(i) in train_index):
                            self.data.append(img)
                            # label = np.zeros(self.num_class)
                            # label[int(each_class)] = 1.0
                            # label = label.tolist()
                            # self.lable.append(label)
                            self.lable.append(int(each_class))
                    else:
                        if (int(i) not in train_index):
                            self.data.append(img)
                            # label = np.zeros(self.num_class)
                            # label[int(each_class)] = 1.0
                            # label = label.tolist()
                            # self.lable.append(label)
                            self.lable.append(each_class)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.transform is not None:
            current_data = self.transform(self.data[index])
        else:
            current_data = self.data[index]
        current_lable = torch.tensor(self.lable[index])
        return current_data,current_lable



def main():
    from torchvision import transforms
    import DataPre as dp
    transforms = transforms.Compose([
        dp.NumpyToTensor()
    ])
    mydataset_train = MyDataset(num_class = 2,is_train = True, transform=transforms)
    # print(len(mydataset_train))
    loader = Data.DataLoader(
        dataset=mydataset_train,      # 数据集
        batch_size=1,                 # mini batch size
        shuffle=True,                 # 是否打乱数据集
        num_workers=1,        
    )
    for step, (batch_x, batch_y) in enumerate(loader): 
        # print('step:',step,'batch_x:',batch_x,'batch_y:',batch_y)
        print(batch_y)
        pass
if __name__ == "__main__":
    main()