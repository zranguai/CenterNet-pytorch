import torch.nn as nn
from models.backbone import resnet50
from models.neck import resnet50_Decoder
from models.head import resnet50_Head


class CenterNet_Resnet50(nn.Module):
    def __init__(self, num_classes=20, pretrained=False):
        super(CenterNet_Resnet50, self).__init__()
        # 512,512,3 -> 16,16,2048
        self.backbone = resnet50(pretrained=pretrained)
        # 16,16,2048 -> 128,128,64
        self.decoder = resnet50_Decoder(2048)
        #   对获取到的特征进行上采样，进行分类预测和回归预测
        #   128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
        #                -> 128, 128, 64 -> 128, 128, 2
        #                -> 128, 128, 64 -> 128, 128, 2
        self.head = resnet50_Head(channel=64, num_classes=num_classes)

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(self.decoder(feat))
