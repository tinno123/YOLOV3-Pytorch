import torch
from torch import nn
import torch.nn.functional as F
from utils.labels import calculate_iou, build_labels
from utils.train_tools import GetGridCenter


class YoloLoss(nn.Module):
    def __init__(self, num_classes=80,image_size=None,stride=None,anchors = None):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.balance = [0.4,1,4]
        self.anchors = None
        self.stride = [32, 16, 8] if stride is None else stride
        self.anchors = torch.tensor([
        [116, 90] ,[156, 198],[373, 326],
        [30, 61],[62, 45], [59, 119],
        [10, 13], [16, 30], [33, 23]
         ])  if anchors is None else anchors
        self.image_size = [416, 416] if image_size is None else image_size

    def init_biases(self,):
        anchors = []

    def clip_by_tensor(self, t, t_min, t_max):
        t = t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result
    def MSELoss(self, pred, target):
        return torch.pow(pred - target, 2)

    def BCELoss(self, pred, target):
        epsilon = 1e-7
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = - target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output
    def forward(self, pred, label):
        """

        :param pred: shape=[B,10647,85]
        :param label:  shape=[B,10647,10] id,tx,ty,tw,th,pos_mask,xtrue,yture,wtrue,htrue
        :return:
        """
        batch_size = pred.shape[0]
        class_true = label[:, :, 0]  # [batch, 10647]
        x_offset_true = label[:, :, 1]  # x偏移量
        y_offset_true = label[:, :, 2]  # y偏移量
        w_offset_true = label[:, :, 3]  # w偏移量(log)
        h_offset_true = label[:, :, 4]  # h偏移量(log)
        positive_mask = label[:, :, 5].bool()  # 正样本mask
        box_loss_scale = label[:, :, -1]

        # 解析预测
        x_offset_pred = torch.sigmoid(pred[:, :, 0])  # x偏移量预测( 0 -1 )
        y_offset_pred = torch.sigmoid(pred[:, :, 1])  # y偏移量预测
        w_offset_pred = pred[:, :, 2]  # w偏移量预测
        h_offset_pred = pred[:, :, 3]  # h偏移量预测
        conf_pred = pred[:, :, 4]  # 置信度预测
        class_pred = pred[:, :, 5:5 + self.num_classes]  # 类别预测

        if  positive_mask.sum() == 0:
            zero = torch.tensor(0., device=pred.device, requires_grad=True)
            return pred.sum() * 0.0, zero, zero, zero, zero
        
        # 计算每个特征层的索引范围
        layer = []
        start_idx = 0
        for stride in self.stride:
            h, w = self.image_size[0] // stride, self.image_size[1] // stride
            num_anchors = h * w * 3
            layer.append([start_idx, start_idx + num_anchors])
            start_idx += num_anchors


        lbox = torch.zeros(1,device=pred.device)
        lobj = torch.zeros(1,device=pred.device)
        lcls = torch.zeros(1,device=pred.device)

        for i in range(len( layer)):
            #---------------------#
            # 获取该尺度尺度的范围
            # --------------------#
            beigin ,end= layer[i][0], layer[i][1]

            # --------------------
            # 获取该尺度正负样本掩码
            # ---------------------#
            ignore_mask_temp = self.get_ignore(i,pred[:, beigin:end,:],label)
            positive_mask_temp = positive_mask[:, beigin:end].bool().clone()
            box_loss_scale_temp = box_loss_scale[:, beigin:end][positive_mask_temp]
            valid_neg_mask = ((~positive_mask_temp.bool()) * ~ignore_mask_temp).bool()

            # ---------------------#
            # 初始化损失
            # ---------------------#
            xloss = yloss = wloss = hloss = lpos=class_loss=torch.zeros(1,device=pred.device)

            # ---------------------#
            # 获取该尺度的对应正样本标签
            # ---------------------#
            x_offset_true_temp = x_offset_true[:, beigin:end]
            y_offset_true_temp = y_offset_true[:, beigin:end]
            w_offset_true_temp = w_offset_true[:, beigin:end]
            h_offset_true_temp = h_offset_true[:, beigin:end]
            class_true_temp = class_true[:, beigin:end]

            # ---------------------#
            # 获取该尺度的预测信息
            # ---------------------#
            x_offset_pred_temp = x_offset_pred[:, beigin:end]
            y_offset_pred_temp = y_offset_pred[:, beigin:end]
            w_offset_pred_temp = w_offset_pred[:, beigin:end]
            h_offset_pred_temp = h_offset_pred[:, beigin:end]
            conf_pred_temp = conf_pred[:, beigin:end]
            class_pred_temp = class_pred[:, beigin:end]


            # ===位置坐标损失===
            if positive_mask_temp.sum() != 0:
                xloss = torch.mean(self.MSELoss(x_offset_pred_temp[positive_mask_temp], x_offset_true_temp[positive_mask_temp],
                                   ) * box_loss_scale_temp)

                yloss =  torch.mean(self.MSELoss(y_offset_pred_temp[positive_mask_temp], y_offset_true_temp[positive_mask_temp],
                                   )* box_loss_scale_temp)

                wloss = torch.mean(self.MSELoss(w_offset_pred_temp[positive_mask_temp], w_offset_true_temp[positive_mask_temp],
                                  )* box_loss_scale_temp)

                hloss = torch.mean( self.MSELoss(h_offset_pred_temp[positive_mask_temp], h_offset_true_temp[positive_mask_temp],
                                  )* box_loss_scale_temp)
                # ===类别损失===
                class_target = F.one_hot(class_true_temp.long(), num_classes=self.num_classes).float()
                class_loss = F.binary_cross_entropy_with_logits(class_pred_temp[positive_mask_temp],
                                                                class_target[positive_mask_temp], reduction='mean')
                # ===置信度损失===
                lpos = F.binary_cross_entropy_with_logits(conf_pred_temp[positive_mask_temp],
                                                          torch.ones_like(conf_pred_temp[positive_mask_temp]),
                                                          reduction='none')

            # ===置信度损失===
            lneg = F.binary_cross_entropy_with_logits(conf_pred_temp[valid_neg_mask], torch.zeros_like(conf_pred_temp[valid_neg_mask]), reduction='none')

            # ===正负样本置信度损失===
            if positive_mask_temp.sum() != 0:
                obj = torch.mean(torch.cat([lpos, lneg] ,dim=0))
            else :
                obj = torch.mean(lneg)



            # ===loss累加 ===
            lbox += (xloss + yloss + wloss + hloss) * 0.1

            lobj += obj * self.balance[i] * 5
            lcls += class_loss  *0.25


        lbox = lbox * 0.05
        lobj = lobj * 1.0
        lcls = lcls * 1.0


        total_loss = (lbox  + lobj  + lcls)
        num_target = positive_mask.sum().item()
        num_neg = valid_neg_mask.sum().item()

        return total_loss, lbox, lobj, lcls,num_target


    def get_ignore(self, l,pred, label):
        bs = pred.shape[0]
        # ---------------------------#
        # 获取该尺度的下采样步长
        # ---------------------------#
        stride = self.stride[l]

        # ------------------------#
        # 获取特征图大小
        # ------------------------#
        f_w , f_h = self.image_size[0] //  stride , self.image_size[1] // stride

        # ------------------------#
        # 获取每一个位置的下采样步长
        # ------------------------#
        stride = torch.tensor([ stride], device=pred.device).repeat(f_w * f_h * 3)

        ignore_masks = []
        for i in range(bs):
            pred_i = pred[i]
            label_i = label[i]
            ignore_mask = torch.zeros(f_w * f_h * 3, device=pred.device, dtype=torch.bool)
            # ------------------------#
            #   先验框调整参数
            # ------------------------#
            x = torch.sigmoid(pred_i[..., 0])
            y = torch.sigmoid(pred_i[..., 1])
            w = torch.exp(pred_i[..., 2])
            h = torch.exp(pred_i[..., 3])

            # -------------------------#
            #   调整先验框
            # -------------------------#
            anchors = self.anchors[l*3:( l+1 ) * 3].repeat(f_w*f_h,1)
            center = GetGridCenter([f_w, f_h] , 3,stride[0].item(),mode="detect")
            anchors = torch.concat([center, anchors], dim=-1).to(pred.device)

            x = (x + anchors[...,0]) * stride
            y = (y + anchors[...,1]) * stride
            w = (w * anchors[...,2])
            h = (h * anchors[...,3])

            # -------------------------#
            #   获取预测框和真实目标
            # -------------------------#
            boxes = torch.stack([x, y, w, h], dim=-1)
            targets = label_i[label_i[...,5] == 1][:,[6,7,8,9]]
            targets[:, [0, 2]] = targets[:, [0, 2]] * self.image_size[0]
            targets[:, [1, 3]] = targets[:, [1, 3]] * self.image_size[1]

            # -------------------------#
            #   计算iou
            # -------------------------#
            iou = calculate_iou(boxes.unsqueeze(1), targets.unsqueeze(0))

            # -------------------------#
            #  找出大于0.5的地方视为忽略样本
            # -------------------------#
            ignore_index = torch.where(torch.sum(iou > 0.5, dim=-1) > 0)[0]
            ignore_mask[ignore_index] = True
            ignore_masks.append(ignore_mask)



        return torch.stack(ignore_masks)












if __name__ == "__main__":
    loss = YoloLoss(num_classes = 20)
    pred = torch.randn((2,10647,25))

    targets = torch.tensor([
        [0, 300 / 416, 300 / 416, 373 / 416, 236 / 416],
        [1, 200 / 416, 100 / 416, 156 / 416, 198 / 416],
        [0, 34 / 416, 300 / 416, 62 / 416, 45 / 416],
        [1, 90 / 416, 100 / 416, 33 / 416, 23 / 416],
    ]
    )
    size = torch.tensor([[13, 13], [26, 26], [52, 52]])  # [3,2]代表3个特征图的形状
    anchors_config = torch.tensor([
        [116, 90], [156, 198], [373, 326],
        [30, 61], [62, 45], [59, 119],
        [10, 13], [16, 30], [33, 23]
    ])

    labels = build_labels([416, 416], targets, anchors_config).unsqueeze(0).repeat(2,1,1)

    loss_value = loss(pred, labels)
    print(loss_value[0].item())








