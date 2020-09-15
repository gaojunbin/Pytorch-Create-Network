# Pytorch 学习笔记

2020.6

Gao Junbin

---

## 写在前面

稍微得空，整理或者说记录一下pytorch学习过程中的心得与体会。小白的学习笔记，大神请慎入！

本文暂时只记录了部分常用函数，若需要更具体说明，请访问pytorch官网或者阅读[中文文档](https://pytorch-cn.readthedocs.io/zh/latest/)。

本文主要讲解pytorch完成深度学习的步骤和程序实现，对一些原理不会过多深入，若需学习原理，请参考其他理论性教材，如斯坦福大学公开课等。

## 关于yaml文件参数设置

其实这一点无关pytorch，甚至无关深度学习。我们在较大一些的工程项目中，可能经常会存在较大的参数量，如果任由其渗透在工程文件的边边角角，我们修改参数时需要各个文件寻找，很有可能发生遗漏，甚至可能会一不小心造成误操作，改动了某个参数引发一些潜在的bug，这是非常危险的。为了有效避免这种情况，我们习惯性地将必要调整的参数整合在一个文件中，在这个文件中统一进行修改，甚至可以通过选择读取不同文件的不同参数完成不同的任务。那么，关于参数的读取，就是本节的内容。

+ 关于YAML

  YAML 是专门用来写配置文件的语言，非常简洁和强大，远比 JSON 格式方便。（摘自阮一峰先生的博客）

  其基本规则如下：

  > + 大小写敏感
  > + 严格缩进表示层级关系
  > + 缩进时不允许使用tab，只允许使用空格（这一点存疑，来自网络，不过个人使用tab好像也可以）
  > + 缩进的空格数不重要，主要相同层级的元素左对齐即可
  > + ‘#’表示注释，注释该行（同python）

  示例：

  ```yaml
  common:
      workers: 4
      batch_size: 32
  ```

## 后台运行

一般我们的训练文件都会在服务器上跑，但是我们会遇到本地连接服务器后运行train后本地电脑关闭远程就会导致训练中断的问题，这就需要用到tmux这个神器了。具体操作方式可以参考这个[博客](http://www.ruanyifeng.com/blog/2019/10/tmux.html).这里我仅说明安装方法以及最基础的使用方法。

+ 安装

```shell
# Ubuntu 或 Debian
sudo apt-get install tmux

# CentOS 或 Fedora
sudo yum install tmux

# Mac
brew install tmux
```

+ 最简操作流程

```shell
# 新建会话 my_session
tmux new -s $my_session
# 在tmux窗口运行所需的程序
# 按下快捷键Ctrl+b d将会话分离
# 按下快捷键Ctrl+b s列出所有会话
# 下次使用时，重新连接到会话
tmux attach-session -t my_session
```

## python文件中参数的获取

+ 引入必要的包

```python
import argparse      # 命令行参数解析包
import yaml          # yaml文件解析包
```

+ 核心语句示例

```python
parser = argparse.ArgumentParser()			# 创建对象
parser.add_argument('--config', default='config.yaml', type=str)   # 添加参数
args = parser.parse_args()  # 解析添加的参数
```

+ 参数的读取与打印示例

```python
with open(args.config) as f:
		config = yaml.load(f)
print("\n**************************")
for k, v in config['common'].items():
    setattr(args, k, v)
    print('\n[%s]:'%(k), v)
print("\n**************************\n")

batch_size = args.batch_size   # 通过这种方式获取参数
```

+ 命令行调用示例

```shell
python train.py --config config.yaml
```

## pytorch与其他数据互转

+ pytorch与numpy数据互转

```python
import torch
torch_data = torch.from_numpy(numpy_data)
numpy_data = torch_data.numpy()
```

+ pytorch与python普通类型数据互转

```python
import torch
tensor = torch.FloatTensor(data)
```

## pytorch定义variable计算梯度

```python
import torch
variable = Variable(tensor, requires_grad=True)
v_out = torch.mean(variable*variable)
v_out.backward()
```

## pytorch激活函数

```python
import torch.nn.functional as F
torch.relu()
torch.sigmoid()
torch.tanh()
F.softplus()
torch.softmax()
```

## pytorch损失函数

```python
loss_func = nn.CrossEntropyLoss()
loss_func = nn.MSELoss()
……
```

## pytorch优化器

```python
import torch
opt_SGD         = torch.optim.SGD(net.parameters(), lr=LR)
opt_Momentum    = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.8)
opt_RMSprop     = torch.optim.RMSprop(net.parameters(), lr=LR, alpha=0.9)
opt_Adam        = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
……
```

## pytorch网络常用到的基本模块

```python
torch.nn.Linear(in_features, out_features, bias=True)
torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)  
torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)   # padding=(kernel_size-1)/2 if stride=1  --> 卷积后尺寸不变化
torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
torch.nn.ReLU()
torch.nn.Dropout(p=0.5, inplace=False)
……
```

## pytorch定义网络基本步骤

```python
import torch
import torch.nn.functional as F

# way1
class Net(torch.nn.Module):   # 自己定义的网络都需要继承Module类，并实现init方法和forward方法
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   
        self.out = torch.nn.Linear(n_hidden, n_output)   
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        return x
net = Net(1, 10, 1)
net(x)

# way2
net = torch.nn.Sequential(
    torch.nn.Linear(1, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)
net(x)

# way3 整合以上两种方法，推荐
class Net(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.net1 = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1)
        )
    def forward(self, x):
        output = self.net1(x)
        return output
net = Net()
net(x)
```

## pytorch训练（反向传播）步骤

```python
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)	# 优化器，里面存放网络参数和梯度
loss_func = torch.nn.CrossEntropyLoss()                       	# 定义使用的损失函数模型
for _ in range(train_step):
    out = net(x)            # 网络预测值      
    loss = loss_func(out,y) # 预测值与真实值计算损失函数，返回关于网络参数的函数
    optimizer.zero_grad()   # 清除梯度
    loss.backward()         # 反向传播，计算梯度
    optimizer.step()        # 参数调整，完成一步优化
```

## pytorch网络的保存

```python
# 2 ways to save the net
torch.save(net, 'net.pkl')  # save entire net
torch.save(net.state_dict(), 'net_params.pkl')   # save only the parameters
```

## pytorch网络的恢复

```python
def restore_net():  # restore entire net
    net = torch.load('net.pkl')
    prediction = net(x)

def restore_params():   # restore only the parameters
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )     # 先搭建网络
    net.load_state_dict(torch.load('net_params.pkl'))  # 再恢复参数
    prediction = net(x)
```

## pytorch批量训练

```python
import torch
import torch.utils.data as Data

torch_dataset = Data.TensorDataset(input_tensor, lable_tensor)
loader = Data.DataLoader(
    dataset=torch_dataset,      # 数据集
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # 是否打乱数据集
    num_workers=workers,        
)
for epoch in range(3):   # 训练完整训练集的次数
        for step, (batch_x, batch_y) in enumerate(loader): 
            # train your data...
```

## pytorch使用GPU加速

```python
# 测试集
test_x = test_data.test_data.cuda()
test_y = test_data.test_labels.cuda()

# net
net.cuda()

# train
for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
    		x = x.cuda()
        y = y.cuda()

# prediction
pred_y = net(test_x).cuda().data
```

## 查看GPU使用情况

```shell
nvidia-smi
```

## 指定GPU进行训练

```shell
# 直接在终端中指定(官方建议)
CUDA_VISIBLE_DEVICES=1 python train.py
# python代码中设定
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
```

## pytorch实现一个简单的CNN网络示例

（源码来自莫烦python）

```python
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import os

EPOCH = 1               # 训练多少遍完整的训练集
BATCH_SIZE = 50
LR = 0.001              # learning rate

############# train data and test data start ###########
train_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=True,                                     
    transform=torchvision.transforms.ToTensor(),                               
    download=False,
)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# pick 2000 samples to speed up testing
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.  
test_y = test_data.test_labels[:2000]
############# train data and test data end #############

# Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output

cnn = CNN()
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   
        output = cnn(b_x)               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        
        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
```

## 一些注意点（持续更新）

1. nn.CrossEntropyLoss()的真实标签不是不同类标签的概率，而是真实标签的索引。这与tensorflow是有所不同的。
2. 通过cv2.imread()获取的图片通道顺序与卷积的矩阵形状顺序不一致，需要对获取的数据进行reshape(C,h,w)。