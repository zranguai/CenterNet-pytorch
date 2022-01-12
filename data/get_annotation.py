# 目标: 得到图片的路径和对应的坐标， 用于Dataset的load_annotation里面
import os
import random
import xml.etree.ElementTree as ET
from utils.utils import get_classes

classes_path = 'model_data/voc_classes.txt'
classes, _ = get_classes(classes_path)


def convert_path(lab_path):
    in_file = open(lab_path, encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    label_box = []
    for obj in root.iter('object'):
        difficult = 0
        if obj.find('difficult') != None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if int(difficult) == 1 or cls not in classes:
            continue
        cls_id = classes.index(cls)  # 根据列表内容查索引
        xmlbox = obj.find('bndbox')
        # b写成元组的形式表示不可以改变
        b = (
            int(float(xmlbox.find('xmin').text)),
            int(float(xmlbox.find('ymin').text)),
            int(float(xmlbox.find('xmax').text)),
            int(float(xmlbox.find('ymax').text)),
            cls_id
        )
        label_box.append(b)
    return label_box


def get_annotation(mode='train'):
    VOCdevkit_path = os.path.abspath('VOCdevkit')  # VOCdevkit的绝对路径

    image_labels = []  # 0:存储图片路径 1:[框信息:xmin,xmax,ymin,ymax 类别信息id , 第二个]
    train_path = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main', f'{mode}.txt')
    image_ids = open(train_path, encoding='utf-8').read().strip().split()  # 读取每一行，去除空白 切成列表
    for image_id in image_ids:
        img_path = os.path.join(VOCdevkit_path, 'VOC2007/JPEGImages', f'{image_id}.jpg')
        lab_path = os.path.join(VOCdevkit_path, 'VOC2007/Annotations', f'{image_id}.xml')
        label_box = convert_path(lab_path)
        image_labels.append([img_path, label_box])
    return image_labels


if __name__ == '__main__':
    """
    test
    """
    image_label = get_annotation()
    print(image_label)
