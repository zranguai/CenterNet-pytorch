from torch.utils.data import Dataset
from data.get_annotation import get_annotation

import math
import cv2
import numpy as np
from PIL import Image
from utils.utils import cvtColor, preprocess_input


def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


class CenterDataset(Dataset):
    def __init__(self, input_shape, num_classes, train):
        super(CenterDataset, self).__init__()
        self.input_shape = input_shape
        self.output_shape = (int(input_shape[0]/4), int(input_shape[1]/4))
        self.num_classes = num_classes
        self.train = train
        self.annotation_lines = get_annotation(mode=train)

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, item):
        # get_annotation
        # step1: 读取图片和框，类别
        image, box = self.get_data(self.annotation_lines[item], self.input_shape, random=False)
        # step2: 配置正样本 最后出来的尺寸是(128,128)的
        # stage1: 先将框从(512, 512)尺寸全部缩放到(128, 128)尺寸的， {小于0/超出128尺寸的进行裁剪)
        # stage2: 根据图片的中心点绘制高斯半径(中间点为1，其他点依次递减)
        # stage3: 绘制类别的正样本，后面采用focal loss计算损失
        # stage4: 绘制宽高的正样本，后面采用reg_l1_loss计算损失
        # stage5: 绘制中心店偏移损失， 后面采用reg_l1_loss计算损失
        # stage6: 绘制mask
        batch_hm = np.zeros((self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        batch_wh = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg = np.zeros((self.output_shape[0], self.output_shape[1], 2), dtype=np.float32)
        batch_reg_mask = np.zeros((self.output_shape[0], self.output_shape[1]), dtype=np.float32)

        # stage1: 缩放裁剪
        if len(box) != 0:
            boxes = np.array(box[:, :4], dtype=np.float32)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]] / self.input_shape[1] * self.output_shape[1], 0, self.output_shape[1] - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]] / self.input_shape[0] * self.output_shape[0], 0, self.output_shape[0] - 1)

        # 制作正样本
        for i in range(len(box)):
            bbox = boxes[i].copy()
            cls_id = int(box[i, -1])

            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                # stage2
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))

                # 计算中心点坐标
                ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                # stage3 绘制高斯热力图
                batch_hm[:, :, cls_id] = draw_gaussian(batch_hm[:, :, cls_id], ct_int, radius)
                # stage4 绘制宽高
                batch_wh[ct_int[1], ct_int[0]] = 1. * w, 1. * h
                # stage5 绘制中心点偏移量
                batch_reg[ct_int[1], ct_int[0]] = ct - ct_int
                # stage6 绘制mask
                batch_reg_mask[ct_int[1], ct_int[0]] = 1

        image = np.transpose(preprocess_input(image), (2, 0, 1))  # 后面进行predict也要同样处理
        return image, batch_hm, batch_wh, batch_reg, batch_reg_mask

    # 该段代码的主要是: 1. 读取图片， 2.缩放图片到指定大小， 3. 缩放/平移框
    def get_data(self, annotation_line, input_shape=[512, 512], random=False):
        line = annotation_line
        # 读取图片并转换成RGB图像,防止灰度图报错
        image = Image.open(line[0])
        image = cvtColor(image)
        # 获取图像宽高和目标的宽高
        iw, ih = image.size
        w, h = input_shape
        # 获取预测框,转换成numpy类型(传入进来的是xmin, ymin, xmax, ymax, cls)
        box = np.array([np.array(box_t) for box_t in line[1]])
        if not random:  # 不进行数据增强
            # 将原始的图片缩放成512*512大小的， 缩放原则为其中较长的缩放到512，例外一条边同理缩放(不一定是512)
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)  # 缩放后尺寸
            nh = int(ih*scale)
            dx = (w-nw)//2  # 缩放之后偏移量
            dy = (h-nh)//2  # 他这个是算的其中一个边的

            # 将图片多余的部分加上灰条，缩放到512的先不管，缩放到其他尺寸的上下(或左右)两边进行填充
            image = image.resize((nw, nh), Image.BICUBIC)  # 把其中一条边resize到512
            new_image = Image.new('RGB', (w, h), (128, 128, 128))  # 512*512全部填上(128, 128, 128)
            new_image.paste(image, (dx, dy))  # 例如对于nw=512,nh=340, w不变，h这里进行上下两个进行填充
            image_data = np.array(new_image, np.float32)  # 转换成np.float32类型 (512, 512, 3)

            # 对框进行调整
            if len(box) > 0:
                np.random.shuffle(box)  # 将box的顺序进行调整
                box[:, [0, 2]] = box[0:, [0, 2]]*nw/iw + dx  # xmin,xmax缩放(应为图片缩放了)+平移(如果上下/左右加灰边需要平移)
                box[:, [1, 3]] = box[0:, [1, 3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0  # 判断xmin,ymin如果小于0就置于0
                box[:, 2][box[:, 2] > w] = w  # xmax超过w的置于w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # 如果不符合要求，就把该框设置为空
            return image_data, box


# DataLoader中collate_fn使用
def centernet_dataset_collate(batch):
    imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks = [], [], [], [], []

    for img, batch_hm, batch_wh, batch_reg, batch_reg_mask in batch:
        imgs.append(img)
        batch_hms.append(batch_hm)
        batch_whs.append(batch_wh)
        batch_regs.append(batch_reg)
        batch_reg_masks.append(batch_reg_mask)

    imgs = np.array(imgs)
    batch_hms = np.array(batch_hms)
    batch_whs = np.array(batch_whs)
    batch_regs = np.array(batch_regs)
    batch_reg_masks = np.array(batch_reg_masks)
    return imgs, batch_hms, batch_whs, batch_regs, batch_reg_masks
