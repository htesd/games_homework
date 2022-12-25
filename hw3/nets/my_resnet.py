import time

import torch
from torch import nn
from torch.nn import functional as F
# class my_resnet(torch.nn.Module):
#

#
class my_res_block(nn.Module):
    def __init__(self,in_channels,out_channels,down_sample=False):
        super().__init__()

        #保证输入输出不变
        self.down=down_sample

        self.channel=out_channels

        self.blocks=nn.Sequential(   nn.Conv2d(kernel_size=(3,3),in_channels=in_channels,out_channels=out_channels,padding=1,stride=1),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU())



        if down_sample:

            self.blocks.add_module(f'Conv_out{out_channels}',
                                   nn.Conv2d(kernel_size=(3, 3), in_channels=out_channels, out_channels=out_channels,
                                             padding=1, stride=2), )
            self.blocks.add_module('BachNorm', nn.BatchNorm2d(out_channels))


            self.onexone=nn.Conv2d(kernel_size=(1,1),in_channels=in_channels,out_channels=out_channels,
                                 padding=0,stride=2)
        else:

            self.blocks.add_module(f'Conv_out{out_channels}',
                nn.Conv2d(kernel_size=(3, 3), in_channels=out_channels, out_channels=out_channels, padding=1, stride=1),)
            self.blocks.add_module('BachNorm',nn.BatchNorm2d(out_channels))

    def forward(self,x):

        Y=self.blocks(x)

        if self.down:

            return F.relu(Y+self.onexone(x))

        else:

            return F.relu(Y+x)


class my_resnet_18(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block=nn.Sequential(



            my_res_block(3,128,down_sample=True),
            my_res_block(128,128),

            my_res_block(128,256,down_sample=True),
            my_res_block(256,256),

            my_res_block(256,512,down_sample=True),
            my_res_block(512,512),

            nn.AdaptiveAvgPool2d(output_size=(1,1)),

            nn.Flatten(),

            nn.Linear(512,10)


        )

    def forward(self,x):
        return self.block(x)






# #事实证明当除不尽的时候会往上取整
res_net=my_resnet_18()

ten=torch.randn([10,3,32,32],device='cuda:0')
res_net.to('cuda:0')

for i in range(1000):
    T1 = time.perf_counter()
    print(res_net(ten).shape)

    T2 = time.perf_counter()
    print('程序运行时间:%s毫秒' % ((T2 - T1) * 1000))
