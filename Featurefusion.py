import torch 
from torch import nn


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
       
        self.relu1 = nn.ReLU()
        self.fc1   = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.fc2   = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.fc3   = nn.Conv2d(in_planes, in_planes, 1, bias=False)

        self.ln1 = nn.Linear(in_planes,in_planes,bias=False)
        self.ln2 = nn.Linear(in_planes,in_planes/2,bias=False)
        self.ln3 = nn.Linear(in_planes/2,in_planes,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.fc1(x)
        x= self.relu1(x)
        x1 = self.fc2(x)
        x= self.relu1(x)
        x2 = self.fc3(x)
        x= self.relu1(x)
        avg_out = self.ln3(self.relu1(self.ln2(self.relu1(self.ln1(self.avg_pool(x1))))))
        max_out = self.ln3(self.relu1(self.ln2(self.relu1(self.ln1(self.max_pool(x2))))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self,in_planes,ratio=1,kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.relu1 = nn.ReLU()
        self.fc1   = nn.Conv2d(in_planes/ratio, in_planes, 1, bias=False)


        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x= self.relu1(x)
    
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Fea_fusion_block(nn.Module):
    def __init__(self, channel,in_planes=64, kernel_size=7):
        super(Fea_fusion_block, self).__init__()

        self.fc1_1   = nn.Conv2d(in_planes, in_planes, 3, bias=False)
        self.fc1_2   = nn.Conv2d(in_planes, in_planes, 3, bias=False)

        self.conv1 = nn.Conv2d(in_planes,in_planes,3,bias=False)
        self.conv2 = nn.Conv2d(in_planes,in_planes,3,bias=False)
        self.conv3 = nn.Conv2d(in_planes*2,in_planes*2,3,bias=False)
        self.conv4 = nn.Conv2d(in_planes*2,in_planes*2,3,bias=False)
        self.conv5 = nn.Conv2d(in_planes*2,in_planes,3,bias=False)
        self.conv6 = nn.Conv2d(in_planes,in_planes,3,bias=False)
        self.conv7 = nn.Conv2d(in_planes,in_planes,3,bias=False)
        self.relu = nn.ReLU()
        self.channelattention = ChannelAttention(channel)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, fa,fb):
        fa = self.fc1_1(fa)
        fa = self.relu(fa)
        fa = self.fc1_2(fa)
        fa = self.relu(fa)
        fb = self.relu(self.conv1(fb))
        fb = self.relu(self.conv2(fb))
        fb = fb * self.channelattention(fa)
        fb = fb * self.spatialattention(fa)
        fb= torch.concat(fb,fa)

        out=self.relu(self.conv3(fb))
        out=self.relu(self.conv4(out))
        out=self.relu(self.conv5(out))

        out=torch.concat(out,fb)

        out=self.relu(self.conv6(out))
        out=self.relu(self.conv7(out))

        return out
