import colorsys
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import ImageDraw, ImageFont

from utils.utils import cvtColor, get_classes, preprocess_input, resize_image
from utils.utils_bbox import decode_box, postprocess
from models.centernet import CenterNet_Resnet50


class CenterNet(object):
    def __init__(self):
        self.classes_path = 'model_data/voc_classes.txt'
        self.model_path = 'model_data/centernet_resnet50_voc.pth'
        self.confidence = 0.3
        self.nms_iou = 0.3
        self.input_shape = [512, 512]
        self.cuda = True
        self.letterbox_image = False
        self.nms = True
        self.class_names, self.num_classes = get_classes(self.classes_path)  # 计算总的类的数量

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

    def generate(self):
        """
        载入模型和权重
        :return:
        """
        self.net = CenterNet_Resnet50(num_classes=self.num_classes, pretrained=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print(f'{self.model_path} model, and classes loaded')

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])  # 图片宽高
        image = cvtColor(image)  # 转换成RGB
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # 预处理，归一化 这里和训练时的操作一样
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(image_data)).type(torch.FloatTensor)  # numpy-->tensor
            if self.cuda:
                images = images.cuda()

            # 送进网络进行forward
            outputs = self.net(images)
            # 对网络输出的结果进行decode
            outputs = decode_box(outputs[0], outputs[1], outputs[2], self.confidence, self.cuda)
            # 使用nms取出多余的框并将框缩放到图片的尺寸
            results = postprocess(outputs, self.nms, image_shape, self.input_shape, self.letterbox_image, self.nms_iou)
            # 没有检测出物体，返回原图
            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 5], dtype='int32')
            top_conf = results[0][:, 4]
            top_boxes = results[0][:, :4]

        # 字体边框
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.input_shape[0], 1)
        # 绘制图像
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image
