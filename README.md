# YOLOv3 目标检测

基于 PyTorch 实现的 YOLOv3 目标检测算法。

## 性能指标

| 数据集 | 预训练权重 | mAP@0.5 | 说明                    |
|--------|-----------|---------|-----------------------|
| VOC2012 | DarkNet-53 Backbone | 68.0% | 仅加载DarkNet53主干网络预训练权重 |


权重 : https://pan.baidu.com/s/1lrGKaaRbG5oggukrqjHO5w?pwd=268q 提取码: 268q
### 与参考实现对比

代码架构和复现参考：[bubbliiiing/yolo3-pytorch](https://github.com/bubbliiiing/yolo3-pytorch)

- **训练速度**：相比 bubbliiiing 的实现快约 30%/iteration
- **精度表现**：使用相同权重和超参数下，mAP 无差异（backbone 权重来自 bubbliiiing 仓库）
- **优化点**：改进数据加载、减少冗余计算、更好的内存管理

## 特性

- 支持检测、评估、ONNX 导出三种模式
- 两阶段训练：先冻结主干再微调整个网络
- 混合精度训练（FP16）
- 自动保存检查点和 mAP 评估

## 环境要求

```
torch>=1.13.0
torchvision>=0.14.0
opencv-python>=4.7.0
numpy>=1.24.0
tqdm>=4.65.0
torchmetrics>=0.11.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
```

## 项目结构

```
yolov3_replication/
├── models/                 # 模型定义
│   ├── DarkNet.py         # DarkNet-53 骨干网络
│   ├── FPN.py            # 特征金字塔网络
│   └── YOLOv3.py         # YOLOv3 完整模型
├── utils/                  # 工具函数
│   ├── datasets.py        # 数据加载与增强
│   ├── labels.py          # 标签构建
│   ├── loss.py            # YOLO 损失函数
│   └── train_tools.py     # 训练辅助工具
├── weights/                # 预训练权重(均来自于bubbliiiing仓库)
│   ├── darknet53_backbone_weights.pth
│   └── yolo_weights.pth
├── train.py                # 训练脚本
├── yolo.py                 # 功能封装
├── detect.py               # 检测示例
├── val.py              # mAP 评估示例
└── export.py             # ONNX 导出示例
```

## 使用说明

### 训练

```bash
python train.py
```

编辑 `train.py` 配置：
- 数据集路径（root_dir, images_dir, label_dir）
- 超参数（batch_size, epochs, learning rate）
- 模型设置（num_classes, input_shape）

训练流程：
1. 第 0-50 轮：冻结主干，只训练 FPN 和检测头
2. 第 50-300 轮：解冻所有层，微调整个网络
3. 根据验证集损失自动保存最佳模型
4. 每 10 轮评估一次 mAP

日志和检查点保存在 `runs/train/<timestamp>/`

### 验证

训练期间每 10 轮自动执行验证。

独立评估：
```bash
python val.py
```

编辑 `val.py` 设置：
- `label_path`：验证集标签路径
- `image_path`：验证集图片路径
- `weight`：训练好的模型权重路径

输出 mAP@0.5:0.95, mAP@0.5, mAP@0.75

### 导出

导出训练好的模型为 ONNX 格式：
```bash
python export.py
```

编辑 `export.py` 设置：
- `weight`：训练好的模型权重路径
- `onnx_path`：输出 ONNX 文件路径
- `num_classes`：类别数量
- `image_size`：输入尺寸 [宽, 高]

导出的 ONNX 模型可用于 ONNX Runtime、TensorRT 或 OpenVINO 部署。

## 数据集准备

按 YOLO 格式组织数据集：

```
dataset/
├── images/
│   ├── train/            # 训练集图片
│   ├── val/              # 验证集图片
│   └── test/             # 测试集图片（可选）
└── labels/
    ├── train/            # 训练集标签 (txt)
    ├── val/              # 验证集标签 (txt)
    └── test/             # 测试集标签 (txt，可选)
```

标签格式（每行一个目标）：
```
<类别ID> <中心点x> <中心点y> <宽度> <高度>
```
所有坐标为归一化值 (0-1)，其中 cx, cy 为中心点，w, h 为宽高。

## 配置说明

### 训练参数（train.py）

```python
# 基础设置
device = "cuda"                  # 训练设备
fp16 = True                      # 混合精度训练
seed = 11                        # 随机种子

# 模型设置
input_shape = [416, 416]         # 输入尺寸（必须是32的倍数）
number_classes = 20              # 类别数量

# 超参数
batch_size = 16
epoches = 300
freeze_epoch = 50
Init_lr = 1e-2                   # 初始学习率
optimizer_type = "sgd"
momentum = 0.937
weight_decay = 5e-4
```

### 推理参数（yolo.py）

```python
YOLO(
    num_classes=20,              # 类别数
    score_threshold=0.25,        # 置信度阈值
    iou_threshold=0.5,           # NMS IoU 阈值
    weight="path/to/weights",    # 权重文件路径
    image_size=[416, 416],       # 推理尺寸
    mode="detect"                # 模式: detect/mAP/onnx
)
```



## 注意事项

1. 输入尺寸必须是 32 的倍数（如 320, 416, 512, 608）
2. 类别索引从 0 开始连续编号

## 许可证

MIT License. 详见 LICENSE 文件。

## 参考

- 参考实现:[bubbliiiing/yolo3-pytorch](https://github.com/bubbliiiing/yolo3-pytorch) 
- PyTorch: https://pytorch.org/
- PASCAL VOC: http://host.robots.ox.ac.uk/pascal/VOC/
- YOLOv3: https://arxiv.org/abs/1804.02767
---

最后更新: 2026-06-14  
版本: v1.0  
测试结果: VOC2012 mAP@0.5 = 68.0% (仅加载 Backbone 预训练)
