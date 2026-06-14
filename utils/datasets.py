import os.path

import cv2
import torch,torch.utils.data
from pathlib import Path
import random
import numpy as np
from PIL import Image

from utils.labels import build_labels



def rand( a=0, b=1):
    return np.random.rand() * (b - a) + a
def get_random_data(image, box, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
    """
    Args:
        image: PIL图像
        box: np.array, shape=(N,5), [class_id, xc, yc, w, h] (归一化坐标)
        input_shape: (width, height) 目标尺寸
        jitter: 长宽比扭曲参数
        hue, sat, val: HSV颜色变换参数
        random: 是否进行随机增强
    
    Returns:
        image_data: 增强后的图像 (np.array)
        box: 增强后的bbox (np.array), 格式保持 [class_id, xc, yc, w, h] (归一化)
    """
    # ------------------------------#
    #   获得图像的高宽与目标高宽
    # ------------------------------#
    iw, ih = image.size
    w, h = input_shape
    
    # 深拷贝避免修改原数据
    box = box.copy()
    
    if not random:
        # ------------------------------------------#
        #   验证模式：仅做letterbox填充，不做随机增强
        # ------------------------------------------#
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        # ---------------------------------#
        #   将图像多余的部分加上灰条
        # ---------------------------------#
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.uint8)

        # ---------------------------------#
        #   对真实框进行调整（缩放+平移）
        # ---------------------------------#
        if len(box) > 0:
            # 缩放坐标
            box[:, 1] = box[:, 1] * scale + dx / w  # xc
            box[:, 2] = box[:, 2] * scale + dy / h  # yc
            box[:, 3] = box[:, 3] * scale           # w
            box[:, 4] = box[:, 4] * scale           # h
            
            # 裁剪到合法范围
            box[:, 1] = np.clip(box[:, 1], 0, 0.99)
            box[:, 2] = np.clip(box[:, 2], 0, 0.99)
            box[:, 3] = np.clip(box[:, 3], 0, 0.99)
            box[:, 4] = np.clip(box[:, 4], 0, 0.99)
            
            # 过滤无效框
            box = box[np.logical_and(box[:, 3] > 1/w, box[:, 4] > 1/h)]

        return image_data, box

    # ------------------------------------------#
    #   训练模式：随机增强
    # ------------------------------------------#
    
    # ------------------------------------------#
    #   对图像进行缩放并且进行长和宽的扭曲
    # ------------------------------------------#
    new_ar = iw / ih * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # ------------------------------------------#
    #   将图像多余的部分加上灰条
    # ------------------------------------------#
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # ------------------------------------------#
    #   翻转图像
    # ------------------------------------------#
    flip = rand() < .5
    if flip: 
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    image_data = np.array(image, np.uint8)
    
    # ---------------------------------#
    #   对图像进行色域变换
    # ---------------------------------#
    r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    hue_ch, sat_ch, val_ch = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
    dtype = image_data.dtype
    
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    image_data = cv2.merge((cv2.LUT(hue_ch, lut_hue), 
                            cv2.LUT(sat_ch, lut_sat), 
                            cv2.LUT(val_ch, lut_val)))
    image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

    # ---------------------------------#
    #   对真实框进行调整
    # ---------------------------------#
    if len(box) > 0:
        np.random.shuffle(box)
        
        # 缩放和平移（注意：这里是像素坐标的变换）
        box[:, 1] = box[:, 1] * nw + dx  # xc: 归一化->像素->缩放+平移
        box[:, 2] = box[:, 2] * nh + dy  # yc
        box[:, 3] = box[:, 3] * nw       # w: 按宽度缩放
        box[:, 4] = box[:, 4] * nh       # h: 按高度缩放
        
        # 水平翻转
        if flip:
            box[:, 1] = w - box[:, 1]
        
        # 转换回归一化坐标
        box[:, 1] = box[:, 1] / w
        box[:, 2] = box[:, 2] / h
        box[:, 3] = box[:, 3] / w
        box[:, 4] = box[:, 4] / h
        
        # 裁剪到合法范围
        box[:, 1] = np.clip(box[:, 1], 0, 0.99)
        box[:, 2] = np.clip(box[:, 2], 0, 0.99)
        box[:, 3] = np.clip(box[:, 3], 0, 0.99)
        box[:, 4] = np.clip(box[:, 4], 0, 0.99)
        
        # 过滤无效框
        box_w = box[:, 3]
        box_h = box[:, 4]
        box = box[np.logical_and(box_w > 1/w, box_h > 1/h)]

    return image_data, box






class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir : str, label_dir : str ,images_dir: str,anchors,image_size,mode = 'train'):

        #----------------------#
        # 标签路径
        #----------------------#
        self.labelpath =  Path(root_dir) /  Path( label_dir)

        # ----------------------#
        # 图像路径
        # ----------------------#
        self.imagespath = Path(root_dir) /  Path(images_dir)

        # ----------------------#
        # 模型输入尺寸
        # ----------------------#
        self.image_size= image_size

        # ----------------------#
        # 标签构建所需的Anchors
        # ----------------------#
        self.anchors = anchors

        # ----------------------#
        # 标签构建模式
        # ----------------------#
        self.mode = mode

        # ----------------------#
        # 加载标签
        # ----------------------#
        self.samples = []
        self.load_samples()

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]

        # -------------------------#
        # 获取当前batch所需的图像和标签
        # -------------------------#
        image = Image.open(sample['image_path']).convert('RGB')
        bboxes = self.analysis_label(sample['label_path'])

        # -------------------------#
        # 获取当前batch图像尺寸
        # -------------------------#
        w, h = image.size

        # -------------------------#
        # 进行图像增强
        # -------------------------#
        if self.mode == 'train':
            image, bboxes = get_random_data(image, bboxes, self.image_size, random=True)
        else:
            image, bboxes = get_random_data(image, bboxes, self.image_size, random=False)
        # -------------------------#
        # 重新采样
        # -------------------------#
        if bboxes.shape[0] < 0:
            print('no bbox')
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

        # -------------------------#
        # ToTensor
        # -------------------------#
        image_tensor = self.image2tensor(image)
        bboxes_tensor = self.bboxes2tensor(bboxes)
        #self._save_vis(image_tensor, bboxes_tensor, sample['image_path'])
        # -------------------------#
        # 标签构建
        # -------------------------#
        targets = build_labels(self.image_size, bboxes_tensor, self.anchors)




        return image_tensor , targets






    def image2tensor(self, image):
        image_tensor = torch.from_numpy(np.array(image)).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).contiguous()
        return image_tensor

    def bboxes2tensor(self, bboxes):
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float32) if bboxes.shape[0] > 0 else torch.zeros((0, 5))
        return bboxes_tensor



    def load_samples(self):
        # 遍历标签文件夹，根据标签名称去找对应的图片
        all_files = os.listdir(self.labelpath)
        for file in all_files:
            labelpath = Path(self.labelpath / Path(file))
            
            if os.path.isfile(labelpath) and file.endswith('.txt'):
                # 检查标签文件是否为空
                bboxes = self.analysis_label(labelpath)
                if len(bboxes) == 0:
                    print(f"Warning: Label file {labelpath} is empty, skipping...")
                    continue
                
                image_filename = file.split('.')[0] + '.jpg'
                imagepath = Path(self.imagespath / Path(image_filename))
                
                # 检查图片文件是否存在
                if not os.path.isfile(imagepath):
                    print(f"Warning: Image file {imagepath} not found, skipping...")
                    continue
                
                self.samples.append({
                    'image_path': imagepath,
                    'label_path': labelpath
                })

    def analysis_label(self, label_path):
        bboxes = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    # 取消类别ID限制，接受所有合法的类别ID（>=0）
                    if class_id >= 0:
                        center_x, center_y, width, height = map(float, parts[1:5])
                        bboxes.append([class_id, center_x, center_y, width, height])
        return np.array(bboxes) if len(bboxes) > 0 else np.zeros((0, 5))



    def _save_vis(self, img_tensor, bboxes, src_path, tgt_size=416, save_dir='runs/vis'):
        """
        把 tensor 画框后存成 jpg，方便肉眼检查坐标是否正确。
        输入：
            img_tensor : C×H×W  0-1
            bboxes     : [[cls,cx,cy,w,h], ...]  归一化
            src_path   : 原图路径（仅用于命名）
            tgt_size   : 画布尺寸
        """
        import os, cv2
        os.makedirs(save_dir, exist_ok=True)

        # tensor -> numpy RGB
        img = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if len(bboxes) < 0:
            print('没有标注信息无法画框')

        # 画框
        for cls, cx, cy, w, h in bboxes:
            if cx < 0 or cy < 0 or w >= 1 or h >= 1 or w <= 0 or h <= 0 or cx >= 1 or cy >= 1:
                print('标注信息超出范围')
                print(cx, cy, w, h)
            x1 = int((cx - w / 2) * tgt_size)
            y1 = int((cy - h / 2) * tgt_size)
            x2 = int((cx + w / 2) * tgt_size)
            y2 = int((cy + h / 2) * tgt_size)


            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{int(cls)}', (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        save_name = os.path.basename(src_path).replace('.jpg', '_vis.jpg')
        from pathlib import Path
        save_path = Path(save_dir) / save_name
        cv2.imwrite(str(save_path), img)
