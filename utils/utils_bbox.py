import numpy as np
import torch
import torch.nn as nn
from torchvision.ops import nms


def pool_nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def decode_box(pred_hms, pred_whs, pred_offsets, confidence, cuda):
    """
    将网络出来的热力图，宽高，偏移 根据给定的confidence进行过滤
    Parameters:
    :param pred_hms: 预测的热力图/类别图 shape=(1, 20, 128, 128)
    :param pred_whs: 预测的宽高 shape=(1, 2, 128, 128)
    :param pred_offsets: 预测的offset shape=(1, 2, 128, 128)
    :param confidence: 类别置信度 default=0.3
    :param cuda:

    :return: [bboxes(归一化), class_conf, class_pred]
    """
    pred_hms = pool_nms(pred_hms)
    b, c, output_h, output_w = pred_hms.shape  # 1, 20, 128, 128
    detects = []
    # 对于predict, 只传入一张图片
    for batch in range(b):
        # heat_map: 128*128, num_classes
        # pred_wh: 128*128, 2
        # pred_offset: 128*128, 2
        heat_map = pred_hms[batch].permute(1, 2, 0).view([-1, c])  # 这里的permute(1, 2, 0)其实跟训练的时候是一样的
        pred_wh = pred_whs[batch].permute(1, 2, 0).view([-1, 2])  # 变成128*128， 2(该维度的只没有变)
        pred_offsets = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])

        yv, xv = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w))  # xv: 128*128特征点x轴坐标
        xv, yv = xv.flatten().float(), yv.flatten().float()
        if cuda:
            xv = xv.cuda()
            yv = yv.cuda()
        class_conf, class_pred = torch.max(heat_map, dim=-1)  # 128*128:特征点种类置信度，特征点种类
        mask = class_conf > confidence
        # 取出得分筛选后对应的结果
        pred_wh_mask = pred_wh[mask]
        pred_offsets_mask = pred_offsets[mask]
        if len(pred_wh_mask) == 0:
            detects.append([])
            continue

        # 原先的网格加上偏移量
        xv_mask = torch.unsqueeze(xv[mask] + pred_offsets_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + pred_offsets_mask[..., 1], -1)
        # 计算一半w, h 用于计算左上角， 右下角
        half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
        # 得到左上角，右下角
        bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)
        bboxes[:, [0, 2]] /= output_w  # 相当于128归一化，方便后面缩放到图片尺寸
        bboxes[:, [1, 3]] /= output_h
        detect = torch.cat([bboxes, torch.unsqueeze(class_conf[mask], -1), torch.unsqueeze(class_pred[mask], -1).float()], dim=-1)
        detects.append(detect)
    return detects


def centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    if letterbox_image:
        #   这里求出来的offset是图像有效区域相对于图像左上角的偏移情况
        #   new_shape指的是宽高缩放情况
        new_shape = np.round(image_shape * np.min(input_shape/image_shape))
        offset = (input_shape - new_shape)/2./input_shape
        scale = input_shape/new_shape

        box_yx = (box_yx - offset) * scale
        box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes


def postprocess(prediction, need_nms, image_shape, input_shape, letterbox_image, nms_thres=0.4):
    output = [None for _ in range(len(prediction))]
    # 预测一张图片只进行一次
    for i, image_pred in enumerate(prediction):
        detections = prediction[i]
        if len(detections) == 0:
            continue
        # 该预测中有那些类别
        unique_labels = detections[:, -1].cpu().unique()

        if detections.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        # nms
        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            if need_nms:
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4],
                    nms_thres
                )
                max_detections = detections_class[keep]
            else:
                max_detections = detections_class
            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i] = output[i].cpu().numpy()
            box_xy, box_wh = (output[i][:, 0:2] + output[i][:, 2:4]) / 2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4] = centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    return output
