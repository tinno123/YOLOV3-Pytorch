from yolo import YOLO

model = YOLO(
    num_classes=80,
    label_path=r"E:\voc2012\yolo\labels\val",
             image_path=r"E:\voc2012\yolo\images\val",
             score_threshold=0.001,
             weight="weights/yolo_weights.pth",
             mode="mAP")

result = model.get_map()
