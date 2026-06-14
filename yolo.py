import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Union, Dict

import torch
import torch.nn as nn
from tqdm import tqdm
from models.YOLOv3 import YOLOv3
from utils.train_tools import GetGridCenter, load_weights_by_shape, image2tensor
import numpy as np
import cv2
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# ==================== 数据层 ====================
@dataclass
class DetectionResult:
    boxes_xyxy: np.ndarray    # [x1, y1, x2, y2] 像素坐标
    scores: np.ndarray        # 置信度
    class_ids: np.ndarray     # 类别ID
    
    @property
    def xywh(self) -> np.ndarray:

        boxes = self.boxes_xyxy.copy()
        boxes[:, 0] = (self.boxes_xyxy[:, 0] + self.boxes_xyxy[:, 2]) / 2  # cx
        boxes[:, 1] = (self.boxes_xyxy[:, 1] + self.boxes_xyxy[:, 3]) / 2  # cy
        boxes[:, 2] = self.boxes_xyxy[:, 2] - self.boxes_xyxy[:, 0]        # w
        boxes[:, 3] = self.boxes_xyxy[:, 3] - self.boxes_xyxy[:, 1]        # h
        return boxes
    
    @property
    def xywhn(self) -> np.ndarray:

        return self.xywh  # 简化处理
    
    @property
    def xyxyn(self) -> np.ndarray:

        return self.boxes_xyxy  # 简化处理
    
    @property
    def xyxy(self) -> np.ndarray:

        return self.boxes_xyxy
    
    @property
    def score(self) -> np.ndarray:
        return self.scores
    
    def __repr__(self):
        return f"DetectionResult(boxes={len(self.boxes_xyxy)}, scores={self.scores}, class_ids={self.class_ids})"
    
    def __str__(self):
        lines = [f"DetectionResult 对象:"]
        lines.append(f"  boxes_xyxy: {self.boxes_xyxy}")
        lines.append(f"  scores: {self.scores}")
        lines.append(f"  class_ids: {self.class_ids}")
        return "\n".join(lines)


# 兼容性别名
result = DetectionResult

# ==================== 核心推理层 ====================
class YOLOv3Inference:
    
    def __init__(
        self,
        model: nn.Module,
        anchors: torch.Tensor,
        stride: List[int],
        image_size: Tuple[int, int],
        device: torch.device,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.5
    ):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.stride = stride
        
        # 初始化 Anchors
        self._init_anchors(anchors)
    
    def _init_anchors(self, anchors: torch.Tensor):
        anchors = anchors.reshape(3, -1).to(self.device)
        num_grid_per_layer = [
            self.image_size[0] // s * self.image_size[1] // s 
            for s in self.stride
        ]
        
        anchors_per_layer = []
        for i in range(3):
            w, h = self.image_size[0] // self.stride[i], self.image_size[1] // self.stride[i]
            center = GetGridCenter([w, h], 3, self.stride[i], "detect").to(self.device)
            anchor_expanded = (anchors[i].repeat(num_grid_per_layer[i])).reshape(-1, 2)
            anchors_per_layer.append(torch.cat([center, anchor_expanded], dim=-1))
        
        self.anchors = anchors_per_layer
    
    def _decode_predictions(self, predictions: List[torch.Tensor]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """解码模型输出并执行 NMS（私有方法）"""
        bboxes_list = []
        class_ids_list = []
        scores_list = []
        
        for i in range(3):
            prediction = predictions[i].squeeze(0)
            anchors = self.anchors[i]
            
            # 解码中心点
            x = torch.sigmoid(prediction[..., 0])
            y = torch.sigmoid(prediction[..., 1])
            
            # 解码宽高
            w = torch.exp(prediction[..., 2])
            h = torch.exp(prediction[..., 3])
            
            # 映射到原图尺寸
            x = (x + anchors[..., 0]) * self.stride[i]
            y = (y + anchors[..., 1]) * self.stride[i]
            w = w * anchors[..., 2]
            h = h * anchors[..., 3]
            boxes = torch.stack([x, y, w, h], dim=-1)
            
            # 计算置信度
            conf = torch.sigmoid(prediction[..., 4])
            class_probs, class_id = torch.max(torch.sigmoid(prediction[..., 5:]), dim=-1)
            conf = conf * class_probs
            
            # 阈值筛选
            conf_mask = conf >= self.score_threshold
            bboxes_list.append(boxes[conf_mask])
            class_ids_list.append(class_id[conf_mask])
            scores_list.append(conf[conf_mask])
        
        # 合并结果
        bboxes = torch.cat(bboxes_list, dim=0)
        class_ids = torch.cat(class_ids_list, dim=0)
        scores = torch.cat(scores_list, dim=0)
        
        # NMS
        if len(bboxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        bboxes_nms = bboxes.clone()
        bboxes_nms[..., 0] = bboxes_nms[..., 0] - bboxes_nms[..., 2] / 2
        bboxes_nms[..., 1] = bboxes_nms[..., 1] - bboxes_nms[..., 3] / 2
        bboxes_nms[..., 0] = bboxes_nms[..., 0] + class_ids * self.image_size[0] * 2
        bboxes_nms[..., 1] = bboxes_nms[..., 1] + class_ids * self.image_size[0] * 2
        
        keep = cv2.dnn.NMSBoxes(
            bboxes_nms.cpu().numpy(), 
            scores.cpu().numpy(), 
            self.score_threshold, 
            self.iou_threshold
        )
        
        return (
            bboxes[keep].cpu().numpy(),
            scores[keep].cpu().numpy(),
            class_ids[keep].cpu().numpy()
        )
    
    def _convert_coordinates(
        self, 
        boxes: np.ndarray, 
        scores: np.ndarray, 
        class_ids: np.ndarray,
        original_shape: Tuple[int, int]
    ) -> DetectionResult:
        original_w, original_h = original_shape
        model_w, model_h = self.image_size
        
        # 计算缩放比例
        scale_x = original_w / model_w
        scale_y = original_h / model_h
        
        # 转换为 xyxy 格式并映射到原始图片尺寸
        boxes_xyxy = boxes.copy()
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * scale_x  # x1
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * scale_y  # y1
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * scale_x  # x2
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * scale_y  # y2
        
        return DetectionResult(
            boxes_xyxy=boxes_xyxy,
            scores=scores,
            class_ids=class_ids
        )
    
    def __call__(self, image: np.ndarray) -> DetectionResult:
        with torch.no_grad():
            h, w = image.shape[:2]
            imagetensor = image2tensor(image, self.device)
            predictions = self.model(imagetensor)
            boxes, scores, class_ids = self._decode_predictions(predictions)
            result = self._convert_coordinates(boxes, scores, class_ids, [w, h])
            return result

# ==================== 应用层 ====================
class YOLO(object):
    """YOLOv3 高层应用接口（保持原有接口不变）"""
    
    def __init__(self, **kwargs):
        # 判断是否传入相关模式
        mode_flag = False
        for k, v in kwargs.items():
            if "mode" in k and v in ["detect", "mAP", "onnx"]:
                mode_flag = True
        if not mode_flag:
            print("请输入相关模式 [detect,mAP,onnx]")
        
        # 默认配置
        self._default = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "num_classes": 20,
            "score_threshold": 0.2,
            "iou_threshold": 0.5,
            "weight": None,
            "image_size": [416, 416],
            "anchors": None,
            "stride": None,
            "save_path": "runs/detect"
        }
        
        # 任务模式
        self.mode = kwargs["mode"]
        
        # 初始化对象参数
        self._apply_mode_config(kwargs)
        for k, v in self._default.items():
            setattr(self, k, v)
        
        print(self.image_size)
        
        # 创建推理引擎
        self.yolo = self._create_inference_engine()
    
    def _apply_mode_config(self, kwargs):
        """根据模式应用配置"""
        mode_kwargs = {
            "detect": ["image_path", "save_path"],
            "mAP": ["score_threshold", "image_path", "label_path"],
            "onnx": ["onnx_path"]
        }[self.mode] + list(self._default.keys())
        
        for k, v in kwargs.items():
            if k in mode_kwargs:
                print(k, v)
                self._default[k] = v
    
    def _create_inference_engine(self) -> YOLOv3Inference:
        """创建推理引擎（内部实现）"""
        # 创建模型
        model = YOLOv3(
            num_classes=self.num_classes,
            mode=self.mode
        ).to(self.device).eval()
        
        # 加载权重
        if self.weight is not None and self.weight != "":
            load_weights_by_shape(model, self.weight)
        
        # 默认 Anchors
        default_anchors = torch.tensor([
            [116, 90], [156, 198], [373, 326],
            [30, 61], [62, 45], [59, 119],
            [10, 13], [16, 30], [33, 23]
        ])
        anchors = default_anchors if self.anchors is None else self.anchors
        
        # 默认 Stride
        stride = [32, 16, 8] if self.stride is None else self.stride
        
        # 创建推理引擎
        return YOLOv3Inference(
            model=model,
            anchors=anchors,
            stride=stride,
            image_size=tuple(self.image_size),
            device=self.device,
            score_threshold=self.score_threshold,
            iou_threshold=self.iou_threshold
        )


    def detect(self):
        if not os.path.exists(self.image_path):
            print("图片不存在")
            return
        if os.path.isfile(self.image_path):
            image = cv2.imread(self.image_path)
            result = self.yolo(image)
            boxes = result.xyxy
            id = result.class_ids
            for i in range(len(boxes)):
                cv2.rectangle(image, (int(boxes[i][0]), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])),
                              (0, 255, 0), 2)
                cv2.putText(image, str(id[i]), (int(boxes[i][0] - 5), int(boxes[i][1] - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255), 2)
            cv2.imshow("result", image)
            cv2.waitKey(0)
        elif os.path.isdir(self.image_path):
            print("开始预测文件夹中的图片")
            t = tqdm(total=len(os.listdir(self.image_path)))
            for index, image_name in enumerate(os.listdir(self.image_path)):
                image = cv2.imread(str(Path(self.image_path) / Path(image_name)))
                result = self.yolo(image)
                boxes = result.xyxy
                id = result.class_ids
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    cv2.rectangle(image, (int(x1), int(boxes[i][1])), (int(boxes[i][2]), int(boxes[i][3])),
                                  (0, 255, 0), 2)
                    cv2.putText(image, str(id[i]), (int(boxes[i][0] - 5), int(boxes[i][1] - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 0, 255), 2)
                cv2.imwrite(str(Path(self.save_path) / (f"{index}" + ".jpg")), image)
                t.update(1)
            t.close()

    def get_map(self,model_path=None):
        """
        评估模型的mAP指标
        :return: mAP计算结果
        """
        if model_path:
            load_weights_by_shape(self.yolo.model, model_path)
        print("正在获取所有图片路径.....")
        images_path = self.image_path
        image_path_list = []
        image_shape_list = []

        for image_name in tqdm(os.listdir(images_path), total=len(os.listdir(images_path))):
            image_path_list.append(str(Path(images_path) / Path(image_name)))
            image_shape_list.append(cv2.imread(str(Path(images_path) / Path(image_name))).shape[:2])

        print("正在解析标注文件.....")
        targets = []
        annotation_lists = os.listdir(self.label_path)

        for i, annotationtxt in tqdm(enumerate(annotation_lists), total=len(annotation_lists)):
            annotation_txt_path = Path(self.label_path) / Path(annotationtxt)

            data = []
            with open(annotation_txt_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    data.append([float(i) for i in line.strip().split()])

            data = torch.tensor(data, dtype=torch.float, device=self.device)

            if len(data) == 0:
                targets.append(
                    {"boxes": torch.tensor([], device=self.device), "labels": torch.tensor([], device=self.device)})
                continue

            data[:, [1, 3]] = data[:, [1, 3]] * image_shape_list[i][1]
            data[:, [2, 4]] = data[:, [2, 4]] * image_shape_list[i][0]

            data[:, 1] = data[:, 1] - data[:, 3] / 2
            data[:, 2] = data[:, 2] - data[:, 4] / 2
            data[:, 3] = data[:, 3] + data[:, 1]
            data[:, 4] = data[:, 4] + data[:, 2]
            data = data.int()

            targets.append({"boxes": data[:, 1:5], "labels": data[:, 0]})

        print("正在获取所有预测结果.....")
        preds = []

        for i in tqdm(range(len(image_path_list)), total=len(image_path_list)):
            img = cv2.imread(image_path_list[i])
            result = self.yolo(img)

            boxes = torch.tensor(result.xyxy, device=self.device)
            scores = torch.tensor(result.scores, device=self.device)
            class_id = torch.tensor(result.class_ids, device=self.device)

            preds.append({"boxes": boxes, "labels": class_id, "scores": scores})

        print("正在计算mAP.....")
        metric = MeanAveragePrecision(box_format='xyxy')
        metric.update(preds, targets)
        result = metric.compute()

        print("mAP@0.5:0.95 =", result['map'].item())
        print("mAP@0.5     =", result['map_50'].item())
        print("mAP@0.75    =", result['map_75'].item())


        return result

    def pth2onnx(self):
        input_shape = (1,3,self.image_size[1],self.image_size[0])
        onnx_input = torch.rand(input_shape,device=self.device)
        try:
            torch.onnx.export(
                self.yolo.model,
                onnx_input,  # 示例输入
                self.onnx_path,  # 输出路径
                export_params=True,  # 导出参数
                opset_version=11,  # ONNX算子版本
                do_constant_folding=True,  # 常量折叠优化
                input_names=['input'],
                output_names=['s', 'm', 'l']  # 输出名称
            )
            print(f"onnx 模型导出成功: {self.onnx_path}")
        except Exception as e:
            print(e)



