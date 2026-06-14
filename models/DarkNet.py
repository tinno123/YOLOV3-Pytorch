import torch.nn as nn

class Normallayer(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3, stride =1,padding=1):
        super(Normallayer,self).__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,bias=False,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
    def forward(self ,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leaky(x)
        return x
    
class BottleResBlock(nn.Module):
    def __init__(self,in_channels):
        super(BottleResBlock,self).__init__()

        self.conv1 = Normallayer(in_channels,in_channels//2,kernel_size=1,stride =1,padding = 0)
        self.conv2 = Normallayer(in_channels//2,in_channels,kernel_size=3,stride =1,padding=1)

    def forward(self,x):
        res = self.conv1(x)
        res = self.conv2(res)
        return res + x

class DarkNet_53(nn.Module):
    def __init__(self):
        super(DarkNet_53,self).__init__()
        self.conv1 = Normallayer(3,32)
        self.conv2 = Normallayer(32,64,stride = 2)
        self.conv3_x = self.get_state(1, 64)
        self.conv4  = Normallayer(64,128,stride = 2)
        self.conv5_x = self.get_state(2, 128)
        self.conv6  = Normallayer(128,256,stride = 2)
        self.conv7_x = self.get_state(8,256)
        self.conv8 = Normallayer(256,512,stride = 2)
        self.conv9_x = self.get_state(8,512)
        self.conv10 = Normallayer(512,1024,stride = 2)
        self.conv11_x = self.get_state(4,1024)
    def get_state(self,num_blocks,input_channels):
        model = nn.Sequential()
        for i in range(num_blocks):
            model.add_module( str(i),BottleResBlock(input_channels))

        return  model

    def forward(self,x):
        x = self.conv1(x)  # 416
        x = self.conv2(x)  # 208
        x = self.conv3_x(x)  # 208
        x = self.conv4(x)  # 104
        x = self.conv5_x(x)  # 104
        x = self.conv6(x)  # 52
        x = self.conv7_x(x)  # 52 → route: 52x52
        route1 = x
        x = self.conv8(x)  # 26
        x = self.conv9_x(x)  # 26 → route: 26x26
        route2 = x
        x = self.conv10(x)  # 13
        x = self.conv11_x(x)  # 13 → route: 13x13
        route3 = x
        return  route3, route2, route1



