import torch
from torch.utils.data import DataLoader
from nets import my_resnet
from nets import My_Lenet
import torchvision.transforms as transforms
from util import get_dataset
import  tester
#准备数据

device='cuda:0'

trans=transforms.Compose([transforms.ToTensor(),

                         ])

# train_dataset=get_dataset.dataset_from_image_folder('/home/iiap/桌面/资料/cifar-10/train',trans)
#
# test_dataset=get_dataset.dataset_from_image_folder('/home/iiap/桌面/资料/cifar-10/test',transform=trans)

train_dataset,test_dataset=get_dataset.get_minist_dataset()

train_iter=DataLoader(train_dataset,batch_size=200,shuffle=True,num_workers=8)


test_iter=DataLoader(test_dataset,batch_size=200,shuffle=True,num_workers=8)

#准备网络
net=My_Lenet.MyLenet()
net.to(device=device)
#准备优化器

optim=torch.optim.Adam(net.parameters(),lr=0.1)
Loss=torch.nn.CrossEntropyLoss()
#开始训练

epoch=20




for i in range(epoch):


    for x,y in train_iter:
        net.train()
        optim.zero_grad()

        X=x.cuda(0)
        Y=y.cuda(0)
        Y_hat=net(X)

        loss=Loss(Y_hat,Y)

        #print(loss)

        loss.backward()

        optim.step()

        # with torch.no_grad:
        tester.ImageClassficationTester.caculate(Y,Y_hat,loss)
        # print(Y_hat.shape)
        # #这个地方dim就是实际的shape对应的地方
        # print(torch.argmax(Y_hat, dim=1).shape)

    with torch.no_grad():
        print()
        print("*"*50)

        tester.ImageClassficationTester.get_sum()

        tester.ImageClassficationTester.test_on_testdateset(test_iter,net)
    #print(f"in epoch{i} loss is:{loss_sum/counter:.2f} acc :{acc_sum/(counter*200):.2f}")



