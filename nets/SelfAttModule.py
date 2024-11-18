import torch 
import torch.nn as nn
import DeformConv2d

class Self_Attention(nn.Module):
    def __init__(self, in_planes=64):
        super(Self_Attention, self).__init__()

        self.bn = nn.BatchNorm2d()

        self.fc1   = nn.Conv2d(in_planes, in_planes, 3, bias=False)
        self.fc2   = nn.Conv2d(in_planes, in_planes, 3, bias=False)
        self.fc3   = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.fc4   = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.defc  = DeformConv2d(in_planes, in_planes)

        self.fc5 = nn.Conv2d(in_planes, in_planes, 1, bias=False)

        self.fc6 = nn.Conv2d(in_planes, in_planes/2, 3, bias=False)
        self.fc7 = nn.Conv2d(in_planes/2,in_planes/2,kernel_size=(1,3),stride=2,padding=(0,1),bias=False)
        self.fc8 = nn.Conv2d(in_planes/2,in_planes/2,kernel_size=(3,1),stride=2,padding=(1,0),bias=False)
        self.fc9 = nn.Conv2d(in_planes/2, in_planes/2, 1, bias=False)

        self.fc10 = nn.Conv2d(in_planes, in_planes/2, 3, bias=False)
        self.fc11 = nn.Conv2d(in_planes/2,in_planes/2,kernel_size=(1,5),stride=2,padding=(0,1),bias=False)
        self.fc12 = nn.Conv2d(in_planes/2,in_planes/2,kernel_size=(5,1),stride=2,padding=(1,0),bias=False)
        self.fc13 = nn.Conv2d(in_planes/2, in_planes/2, 1, bias=False)

        self.fc14 = nn.Conv2d(in_planes*2,in_planes, 1, bais=False)

        self.fc15 = nn.Conv2d(in_planes*2,in_planes, 3, bais=False)
        self.relu = nn.ReLU()


    def forward(self, x):

        x1 = self.fc1(x)
        x1 = self.bn(x1)
        x1= self.relu(x1)
        x1 = self.fc2(x1)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x1 = self.fc3(x1)
        x1 = self.bn(x1)
        x1 = self.relu(x1)
        x1 = self.defc(x1)
        x1 = self.relu(x1)
        x1 = self.fc4(x1)
        x1 = self.bn(x1)
        x1 = self.relu(x1)

        x2 = self.fc5(x)
        x2 = self.bn(x2)
        x2 = self.relu(x2)

        x3 = self.relu(self.bn(self.fc6(x2)))
        x3 = self.relu(self.bn(self.fc7(x3)))
        x3 = self.relu(self.bn(self.fc8(x3)))
        x3 = self.relu(self.bn(self.fc9(x3)))

        x4 = self.relu(self.bn(self.fc10(x2)))
        x4 = self.relu(self.bn(self.fc11(x4)))
        x4 = self.relu(self.bn(self.fc12(x4)))
        x4 = self.relu(self.bn(self.fc13(x4)))

        x5 = torch.concat(x4,x3)
        x5 = torch.concat(x5,x2)

        x5 = self.relu(self.bn(self.fc14(x5)))
        out = torch.concat(x5,out)
        
        out = self.relu(self.fc15(out))


        return out
