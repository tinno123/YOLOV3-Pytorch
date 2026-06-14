from yolo import YOLO


model  = YOLO(weight = "",
              mode="onnx",
              num_classes=20,
              onnx_path="yolov3.onnx",
              image_size = [416,416]
              )
model.pth2onnx()