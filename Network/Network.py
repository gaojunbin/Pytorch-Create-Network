# -*- coding: utf-8 -*-
"""
Created on 2020.6
Latest modify 2020.7
@Author: Junbin
@Note  : Network factory
"""
import sys
class Network():
    r"""
    Factory to manage all the networks.
    You should initial with a concrete net.
    """
    def __init__(self,select_net,use_gpu=False):
        """ 
        select_net->str  : select a net
        use_gpu   ->bool : is use gpu or not
        """
        self.select_net = select_net
        if select_net == 'vgg':
            import Network.vgg as vgg
            self.net = vgg.vgg()
        elif select_net == 'alexnet':
            import Network.alexnet as alexnet
            self.net = alexnet.alexnet()
        elif select_net == 'googlenet':
            import Network.googlenet as googlenet
            self.net = googlenet.googlenet()
        elif select_net == 'resnet34':
            import Network.resnet as resnet
            self.net = resnet.resnet34()
        elif select_net == 'my_net':
            import Network.my_net as my_net
            self.net = my_net.my_net()
        else:
            self.net = None
            print('the network name you have entered is not supported yet')
            sys.exit()
        
        if use_gpu:
            self.net = self.net.cuda()
    def get_net(self):
        r"""
        return the net has been created
        """
        return self.net

def main():
    network = Network(select_net='my_net',use_gpu=False)
    net = network.get_net()
    print(net)

if __name__ == '__main__':
    main()