import datetime
import os
from pathlib import Path

import torch

from tqdm import tqdm
from torch.amp import GradScaler
from torch.amp import autocast

from models.YOLOv3 import YOLOv3
from utils.loss import YoloLoss
from utils.train_tools import CosineDecayLR, seed_everything, GetLoader, GetOptimizer, \
    load_weights_by_shape
from yolo import YOLO


def  train(
        model,
        loss,
        optimizer,
        train_loader,
        val_loader,
        device,
        fp16,
        epoches,
        start_epoch,
        freeze_epoch,
        lr_decay,
        mAP_CallBack,
        save_path=''
):

    scaler = GradScaler(device) if fp16 else None
    Freeze_flag = True
    best_valid_loss = 100
    best_valid_map = 0

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = Path(save_path) / Path(timestamp)

    for epoch in range(start_epoch,epoches):
        model.train()



        # --------------------------------#
        #  学习率下降
        # --------------------------------#
        lr = lr_decay(epoch,optimizer)

        # --------------------------------#
        #  模型冻结判断
        # --------------------------------#
        if Freeze_flag and epoch > freeze_epoch:
            model.freeze_backbone(freeze = False)
            print("解冻成功")
            Freeze_flag  =  False



        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{epoches}')
        avg_loss = 0
        train_loss = 0
        valid_loss = 0
        for step, (images,targets) in enumerate(train_loader):
            # --------------------------------#
            # 梯度清零
            # --------------------------------#
            optimizer.zero_grad()

            if fp16:
                with autocast("cuda"):
                    images = images.to(device)
                    targets = targets.to(device)

                    # 前向传播
                    output = model(images)

                    # 计算损失
                    total_loss, lbox, lobj, lcls, num_target = loss(output, targets)

                    # 反向传播
                    scaler.scale(total_loss).backward()

                    # 更新参数
                    scaler.step(optimizer)

                    # 更新缩放因子
                    scaler.update()
            else :
                images = images.to(device)
                targets = targets.to(device)

                # 前向传播
                output = model(images)

                # 计算损失
                total_loss, lbox, lobj, lcls, num_target = loss(output, targets)

                # 反向传播
                total_loss.backward()

                # 更新参数
                optimizer.step()

            avg_loss += total_loss.item()
            pbar.set_postfix({
                'total_loss': avg_loss / (step + 1),
                'lr': lr,
                "pos":(torch.sigmoid(output[..., 4]) < 0.2).sum().item(),
            })
            pbar.update(1)


        pbar.close()
        train_loss = avg_loss / len(train_loader)


        model.eval()
        pbar = tqdm(total=len(val_loader), desc=f'Epoch {epoch + 1}/{epoches}')
        avg_loss = 0

        for step, (images, targets) in enumerate(val_loader):
            with torch.no_grad():
                images = images.to(device)
                targets = targets.to(device)

                # 前向传播
                output = model(images)

                # 计算损失
                total_loss, lbox, lobj, lcls, num_target = loss(output, targets)

                avg_loss += total_loss.item()
            pbar.update(1)

        valid_loss = avg_loss / len(val_loader)

        checkpoint = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': valid_loss
        }
        pbar.close()
        print(f"Train Loss:{train_loss} \n Valid Loss:{valid_loss}")


        os.makedirs(save_path, exist_ok=True)
        torch.save(checkpoint, Path(save_path) / "yolov3.pth")


        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), Path(save_path) / "best_model.pth")
        if (epoch+ 1) % 10 == 0:
            torch.save(model.state_dict(), Path(save_path) / f"epoch_{epoch+1}.pth")
            result= mAP_CallBack.get_map(model.state_dict())
            with open( Path(save_path) / "map.txt", "a") as f:
                map5_95 =  result['map'].item()
                map5 = result['map_50'].item()
                map75 = result['map_75'].item()
                f.write(f"epoch:{epoch+1} map5_95:{map5_95} map5:{map5} map75:{map75}\n")



















if __name__ == "__main__":
    # ---------------------------------#
    #   device  训练使用的设备
    # ---------------------------------#
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------#
    #   Seed    用于固定随机种子
    #           使得每次独立训练都可以获得一样的结果
    # ----------------------------------------------#
    seed = 11
    seed_everything(seed)

    # ----------------------------------------------#
    #   fp16        是否使用混合精度训练
    # ----------------------------------------------#
    fp16 = True

    # ----------------------------------------------#
    #   model_path   模型预训练权重
    #                模型主干默认加载预训练权重，为空则从主干开始训练
    # ----------------------------------------------#
    model_path = ''

    # ----------------------------------------------#
    #   input_shape     输入的shape大小，一定要是32的倍数
    # ----------------------------------------------#
    input_shape = [416, 416]

    # ----------------------------------------------#
    #   start_epoch     训练的起始轮数
    #
    #   epoches         训练总轮数
    #
    #   freeze_epoch    冻结主干训练的轮数
    # ----------------------------------------------#
    start_epoch = 0
    epoches = 300
    freeze_epoch = 50

    # ----------------------------------------------------#
    #  batch_size       训练时每一批次训练的数据数量
    # ----------------------------------------------------#
    batch_size = 16

    # ----------------------------------------------------#
    #   number_classes   类别数量
    # ----------------------------------------------------#
    number_classes = 20

    # ----------------------------------------------------#
    #   Anchors 配置
    # ----------------------------------------------------#
    anchors = torch.tensor([
        [116, 90], [156, 198], [373, 326],
        [30, 61], [62, 45], [59, 119],
        [10, 13], [16, 30], [33, 23]
    ])

    # ----------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ----------------------------------------------#
    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 5e-4

    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr, Min_lr={
        'adam':(1e-3, 1e-6),
        'sgd':(1e-2 / 4, 1e-5 /4)
    }[optimizer_type]

    # ------------------------------------------------------------------#
    #   lr_decay   使用到的学习率下降方式，提供余弦退火
    # ------------------------------------------------------------------#
    lr_decay = CosineDecayLR(Init_lr, Min_lr, epoches)

    # ------------------------------------------------------------------#
    #   num_workers     多线程读取数据核心数量
    # ------------------------------------------------------------------#
    num_workers = 8

    # ----------------------------------------------------#
    #   resume   是否从检查点继续训练
    # ----------------------------------------------------#
    resume = False
    resume_model = ""

    # ----------------------------------------------------#
    #   获取存放数据集根目录
    # ----------------------------------------------------#
    # 数据集存放根目录
    root_dir = r"E:\voc2012\yolo"
    # 数据集路径
    images_dir_train = "images/train"
    images_dir_val = "images/val"
    # 标签路径
    label_dir_train = "labels/train"
    label_dir_val = "labels/val"
    mAP_CallBack = YOLO(label_path=Path( root_dir) / label_dir_val, image_path=Path( root_dir) / images_dir_val,
                 score_threshold=0.001,
                 weight=None, mode="mAP")



    # ----------------------------------------------------#
    #   获取数据加载器
    # ----------------------------------------------------#
    train_loader , val_loader = GetLoader(
        root_dir,
        label_dir_train,
        images_dir_train,
        label_dir_val,
        images_dir_val,
        batch_size,
        num_workers,
        anchors
    )

    # ----------------------------------------------------#
    #   模型训练结果保存路径
    # ----------------------------------------------------#
    save_path = "runs/train"

    # ----------------------------------------------------#
    #   获取模型
    # ----------------------------------------------------#
    model = YOLOv3(number_classes,"weights/darknet53_backbone_weights.pth").to(device)
    if model_path:
        load_weights_by_shape(model, model_path)

    # ----------------------------------------------------#
    #   获取损失函数
    # ----------------------------------------------------#
    loss = YoloLoss(number_classes,anchors = anchors).to(device)

    # ----------------------------------------------------#
    #   获取优化器
    # ----------------------------------------------------#
    optimizer = GetOptimizer(model,optimizer_type,Init_lr,momentum,weight_decay)
    if resume:
        print("从检查点继续训练")
        checkpoint = torch.load(resume_model)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"])


    train(model,loss,optimizer,train_loader,val_loader,device,fp16,epoches,start_epoch,freeze_epoch,lr_decay,mAP_CallBack,save_path=save_path)
















