import os
import time

import numpy as np
import torch
import cv2
import os.path as path
from torch.utils.data.dataset import Dataset
from torchvision import datasets
import torchvision.transforms as T


TEST=False

data_location = "../data"


def num_to_onehot(num,total_num):

    ont_hot=torch.zeros([total_num])
    ont_hot[num]=1
    return ont_hot


def get_minist_dataset(name="mnist"):
    assert name in ['mnist']
    if name == 'mnist':
        train = datasets.MNIST(root=data_location,
                               download=True,
                               transform=T.Compose([

                                   T.ToTensor(),
                                   T.Normalize(0.1370, 0.3081)
                               ]),
                               train=True)
        test = datasets.MNIST(root=data_location,
                              download=True,
                              transform=T.Compose([

                                  T.ToTensor(),
                                  T.Normalize(0.1370, 0.3081)
                              ]),
                              train=False)
    return train, test



class dataset_from_image_folder(Dataset):
    def __init__(self,image_dir, transform=None,use_cuda=False):
        '''
        这里需要初始化一个文件路径和类别相关的表，最好是图像一个，lable一个，这里默认图片的文件夹都是类别
        '''
        self.trans=transform
        #文件路径
        self.image_list=[]
        #类别数量
        self.lable_list=[]

        self.lable_num=0

        if path.isdir(image_dir):

            #这里假设文件夹数量就是类别数量
            self.lable_num=len(os.listdir(image_dir))
            #print(os.listdir(image_dir))

            for i in os.listdir(image_dir):

                file_list=os.listdir(os.path.join(image_dir,i))
                #print(file_list)
                for j in file_list :
                    #把训练路径，分类文件夹和文件名字拼接在一起
                    self.image_list.append(path.join(path.join(image_dir,i),j))

                    self.lable_list.append(int(i))

            print(len(self.image_list))
            print(len(self.lable_list))

        else:
            print("请检查文件路径是否正确")

        return

    def __getitem__(self, index):
        '''
        这个方法就是给定index，然后读取数据和lable，所以我必须初始化列表
        :param index:
        :return:
        '''
        #默认返回numpy方便后续处理

        img=cv2.imread(self.image_list[index])
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        lable=self.lable_list[index]

        #交叉熵损失函数不需要你自己onehot，我觉得其实挺麻烦的

        if self.trans!=None:

            img=self.trans(img)


        return img,lable

    def __len__(self):

        return len(self.image_list)



if TEST:
    dataser=dataset_from_image_folder('/home/iiap/桌面/资料/cifar-10/train')

    data=torch.utils.data.DataLoader(dataser,batch_size=100,shuffle=True,num_workers=16)

    for i,j in data:
        print(i.shape)
        print(j)

