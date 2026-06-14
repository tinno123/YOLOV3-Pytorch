from typing import List

import torch
import torch.nn as nn



class DetectHead(nn.Module):
    def __init__(self,in_channels,num_classes,num_anchors):
        super(DetectHead,self).__init__()
        out_channels = num_anchors * (5 + num_classes)
        self.head_conv = nn.Conv2d(in_channels ,out_channels,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        x= self.head_conv(x)
        return x


class YOLOLayer(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(YOLOLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels * 2, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels * 2)

        self.conv3 = nn.Conv2d(out_channels * 2, out_channels, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(out_channels, out_channels * 2, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_channels * 2)

        self.conv5 = nn.Conv2d(out_channels * 2, out_channels, 1, 1, 0, bias=False)
        self.bn5 = nn.BatchNorm2d(out_channels)

        self.conv6 = nn.Conv2d(out_channels, out_channels * 2, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(out_channels * 2)

        self.leaky = nn.LeakyReLU(0.1)
    def forward(self,x):
        x = self.leaky(self.bn1(self.conv1(x)))
        x = self.leaky(self.bn2(self.conv2(x)))
        x = self.leaky(self.bn3(self.conv3(x)))
        x = self.leaky(self.bn4(self.conv4(x)))
        x = self.leaky(self.bn5(self.conv5(x)))
        route = x
        x = self.leaky(self.bn6(self.conv6(x)))
        return x, route


class UpsampleLayer(nn.Module):
    def __init__(self,in_channels):
        super(UpsampleLayer,self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels//2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(in_channels//2)
        self.leaky = nn.LeakyReLU(0.1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self,shallow_feature,deep_route):
        """
                shallow_feature: 当前层的特征
                 deep_route     : 来自更深层的 route
        """
        deep_route = self.conv(deep_route)
        deep_route = self.bn(deep_route)
        deep_route = self.leaky(deep_route)
        deep_route = self.upsample(deep_route)
        x = torch.cat([deep_route,shallow_feature],dim=1)
        return x



class FPN(nn.Module):
    def __init__(self,channel_list:list,num_classes,module_type="lse"):
        assert len(channel_list) == 3
        super(FPN,self).__init__()

        self.feature_channels_list = channel_list  #存储的是C3,C2,C1的通道数
        self.yolo_input_channels =[] #存储的是构建C3,C2,C1的Yolo层对应的输入通道数
        self.yolo_output_channels = [channel_list[i]//2 for i in range(len(channel_list))] #存储的是构建C3,C2,C1的Yolo层对应的第一层输出通道数
        self.upsample_channels=[] #存储的是构建上采样层的输入大小
        self.C3Layer = None
        self.C3Head = None
        self.C3UpsamplingLayer =None
        self.C2Layer = None
        self.C2Head = None
        self.C1Layer =None
        self.C2UpsamplingLayer = None
        self.C1Head = None
        self.mssa_p3 = None
        self.mssa_p4 = None
        self.mssa_p5 = None

        self.initparameters(num_classes) #初始化相关层参数

        self.count = 0




    def initparameters(self,num_classes):
        YoloLayerList = nn.ModuleList()
        UpsamplingLayerList = nn.ModuleList()
        self.yolo_input_channels.append(self.feature_channels_list[0])
        for i in range(1, len(self.feature_channels_list)):
            self.yolo_input_channels.append(self.feature_channels_list[i] // 2 + self.feature_channels_list[i])
        for i in range(0, len(self.feature_channels_list) - 1):
            self.upsample_channels.append(self.feature_channels_list[i] // 2)
        for i in range(0, len(self.yolo_input_channels)):
            YoloLayerList.append(YOLOLayer(self.yolo_input_channels[i], self.yolo_output_channels[i]))
        for i in range(0, len(self.upsample_channels)):
            UpsamplingLayerList.append(UpsampleLayer(self.upsample_channels[i]))
        self.C3Layer = YoloLayerList[0]
        self.C3Head = DetectHead(1024, num_classes, 3)
        self.C3UpsamplingLayer = UpsamplingLayerList[0]
        self.C2Layer = YoloLayerList[1]
        self.C2Head = DetectHead(512 , num_classes, 3)
        self.C2UpsamplingLayer = UpsamplingLayerList[1]
        self.C1Layer = YoloLayerList[2]
        self.C1Head = DetectHead(256, num_classes, 3)











    def forward(self, darknet_outputs :List[torch.Tensor]):
        # darknet_outputs: [C3, C2, C1] = [13×13×1024, 26×26×512, 52×52×256]
        C3, C2, C1 = darknet_outputs
        self.count =  (self.count + 1 ) % 200
        sp = self.count


        #C3
        out3, route3 = self.C3Layer (C3)
        out3 =  self.C3Head(out3)

        # C3-C2
        up2 = self.C3UpsamplingLayer(C2, route3)
        out2, route2 =  self.C2Layer(up2)
        out2 = self.C2Head(out2)

        # C2-C1
        up1 = self.C2UpsamplingLayer(C1, route2)
        out1, _ =self.C1Layer(up1)
        out1 = self.C1Head(out1)

        return [out3, out2, out1]

if __name__ =="__main__":

    channel_list = [1024,512,256]
    models =FPN(channel_list,20)
    out1 = torch.randn(16,1024,13,13)
    out2 = torch.randn(16, 512, 26, 26)
    out3 = torch.randn(16, 256, 52, 52)
    x = models([out1,out2,out3])
    for i in x:
        print(i.shape)
















