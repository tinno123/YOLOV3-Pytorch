from yolo import YOLO

model  = YOLO(
            num_classes =80,
            score_threshold = 0.25,
            iou_threshold = 0.5,
            image_path = r"street.jpg",
            weight = "weights/yolo_weights.pth",
            mode="detect")
model.detect()

















