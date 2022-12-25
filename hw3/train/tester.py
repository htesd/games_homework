from __future__ import print_function  # 必须在第一行，为了兼容 python 2
import time
import sys
import torch

import numpy as np


class ImageClassficationTester:
    loss_sum = 0

    acc_sum = 0

    lable_num = 0

    epoch=0

    str_len=0

    counter=0
    #在init 里没有静态变量fuck！
    #def __init__(self):


    @staticmethod
    def caculate(y,y_hat,loss):

        ImageClassficationTester.acc_sum += torch.sum(torch.argmax(y_hat, dim=1) == y).item()

        ImageClassficationTester.loss_sum += loss

        ImageClassficationTester.lable_num+=y.shape[0]

        ImageClassficationTester.counter+=1


        if(ImageClassficationTester.str_len!=0):

            sys.stdout.write('\b'*ImageClassficationTester.str_len)
            sys.stdout.flush()


        print("实时训练正确率"+str(ImageClassficationTester.acc_sum/ImageClassficationTester.lable_num),end='')

        ImageClassficationTester.str_len = len("实时训练正确率" + str(ImageClassficationTester.acc_sum / ImageClassficationTester.lable_num) )+1

        #print(ImageClassficationTester.str_len)


    @staticmethod
    def test_on_testdateset(test_iter,net):

        acc_sum=0.0

        test_num=0.0

        cuda=True

        for x,y in test_iter:

            for k in net.state_dict():

                if str(net.state_dict()[k].device)=='cuda:0':
                    cuda=True
                else:
                    cuda=False
                break



            if cuda:

                y_hat=net(x.to('cuda:0'))

                acc_sum += torch.sum(torch.argmax(y_hat, dim=1) == y.to('cuda:0')).item()

            else:
                y_hat = net(x)
                print('cpu')
                acc_sum += torch.sum(torch.argmax(y_hat, dim=1) == y).item()


            test_num+=y.shape[0]

        print(f"测试正确率为{acc_sum/test_num}")

        return  acc_sum/test_num


    @staticmethod
    def get_sum():
        print('')
        print(f"epoch {ImageClassficationTester.epoch} 训练正确率:{ImageClassficationTester.acc_sum/ImageClassficationTester.lable_num} 平均损失:{ImageClassficationTester.loss_sum/ImageClassficationTester.counter}")

        ImageClassficationTester.counter=0

        ImageClassficationTester.lable_num=0

        ImageClassficationTester.acc_sum=0

        ImageClassficationTester.loss_sum=0

        ImageClassficationTester.epoch+=1


if __name__ =='__main__':
    print('Hello\n', end='')
    sys.stdout.write('\b'*2)
    sys.stdout.flush()

