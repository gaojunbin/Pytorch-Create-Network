# -*- coding: utf-8 -*-
"""
Created on 2020.6
Latest modify 2020.7
@Author: Junbin
@Note  : Network factory
"""
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
        # elif select_net == 'resnet':
        #     import Network.resnet as resnet
        #     self.net = resnet()
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
    def train_or_test(self,is_train:bool):
        r"""
        before use the net to calculate, must set is train or not(especially with dropout layer)
        """
        try:
            self.net.train_or_test(is_train)
        except:
            print("\nfailed to set the net model train or test！\n")
            sys.exit()

def main():
    network = Network(select_net='vgg',use_gpu=False)
    net = network.get_net()
    print(net)

if __name__ == '__main__':
    main()