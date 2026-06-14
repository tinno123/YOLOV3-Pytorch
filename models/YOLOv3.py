import torch,torch.nn as nn
from models.DarkNet import DarkNet_53
from models.FPN import FPN
from utils.train_tools import load_weights_by_shape


class YOLOv3(nn.Module):
    def __init__(self,num_classes,load_path ='',mode = "train",module_type="lse"):
        super(YOLOv3,self).__init__()
        self.num_classes = num_classes
        self.channellist = [1024, 512, 256]
        self.load_path = load_path
        self.DarkNet = DarkNet_53()                                    # Backbone :DarkNet_53
        self.FPN = FPN(self.channellist, num_classes=self.num_classes, module_type=module_type) # Near + Head :FPN, YOLODetector
        self.freeze_backbone(freeze=True)
        self.mode =  mode

        if self.load_path:
            load_weights_by_shape(self.DarkNet, load_path)


    def freeze_backbone(self, freeze=True):
        """冻结或解冻 Backbone"""
        for param in self.DarkNet.parameters():
            param.requires_grad = not freeze
        status = "冻结" if freeze else "解冻"
        print(f"DarkNet Backbone {status}成功")

        # 验证冻结状态
        frozen_params = sum(1 for p in self.DarkNet.parameters() if not p.requires_grad)
        total_params = sum(1 for _ in self.DarkNet.parameters())
        print(f"冻结参数: {frozen_params}/{total_params}")


    def forward(self,x):
        r3, r2, r1 = self.DarkNet(x) # Backbone :DarkNet_53
        routes = [r3, r2, r1]
        Multiple_Scale_Features = self.FPN(routes) # Near :FPN
        outputs = []
        for i, feature in enumerate(Multiple_Scale_Features): #Head :YOLODetector
            output = feature.permute(0, 2, 3, 1).contiguous()
            output = output.view(output.size(0), -1, 5 + self.num_classes)
            outputs.append(output)
        final_output = torch.cat(outputs, dim=1)

        if self.mode in ["train"]:
            return final_output
        elif self.mode in ["detect","mAP"]:
            return outputs
        elif  self.mode == "onnx":
            return [ i.permute(0, 2, 3, 1).contiguous()   for i in Multiple_Scale_Features]


if __name__ == '__main__':
    model = YOLOv3(num_classes=80,load_path ='darknet531.pth').to('cuda')
    x = torch.randn(8,3,416,416).to('cuda')
    y = model(x)
    print(y[0].shape)


