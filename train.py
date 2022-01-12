import numpy as np
import torch
import torch.nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from data.centernetDataset import CenterDataset, centernet_dataset_collate
from utils.utils import get_classes
from utils.utils_fit import fit_one_epoch

from models.centernet import CenterNet_Resnet50


def main():
    # 模型设置
    start_epoch = 0
    end_epoch = 10
    batch_size = 1
    Cuda = True
    classes_path = 'model_data/voc_classes.txt'
    input_shape = [512, 512]
    class_names, num_classes = get_classes(classes_path)

    # 准备模型
    model = CenterNet_Resnet50(num_classes, pretrained=True)
    model_train = model.train()
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    # 准备优化器
    optimizer = optim.Adam(model_train.parameters(), lr=1e-3, weight_decay=5e-4)
    # 准备数据集
    train_dataset = CenterDataset(input_shape, num_classes, train='train')
    val_dataset = CenterDataset(input_shape, num_classes, train='val')  # val的数据集

    num_train = len(train_dataset)
    num_val = len(val_dataset)

    epoch_step = num_train // batch_size  # iter数
    epoch_step_val = num_val // batch_size
    # dataloader
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=centernet_dataset_collate
    )
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=centernet_dataset_collate
    )

    # 进行训练  --> epoch
    for epoch in range(start_epoch, end_epoch):
        fit_one_epoch(  # iter
            model_train,  # 训练用
            model,  # 保存模型用
            optimizer,
            epoch,  # 第几个epoch
            epoch_step,  # 需要训练的iter数
            train_dataloader,  # 加载训练数据
            val_dataloader,
            end_epoch,  # 最后一个epoch
            epoch_step_val,
            Cuda,
        )


if __name__ == '__main__':
    main()
