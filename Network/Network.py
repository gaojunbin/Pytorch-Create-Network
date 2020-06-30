class Network():
    def __init__(self,select_net,use_gpu=False):
        """ return given network
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
        return self.net
    def train_or_test(self,is_train):
        try:
            self.net.train_or_test(is_train)
        except:
            print("\n训练网络与测试网络切换失败！\n")
            sys.exit()



def main():
    network = Network(select_net='vgg',use_gpu=False)
    net = network.get_net()
    print(net)

if __name__ == '__main__':
    main()