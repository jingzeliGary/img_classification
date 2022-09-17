'''
VGG:
输入: [3, 224,224 ]
'''

import torch
import torch.nn as nn

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}
# 迁移学习 减去 RGB = [123.68, 116.78, 103.94]

class VGG(nn.Module):
    def __init__(self, feature, num_classes=1000):
        super(VGG, self).__init__()
        self.feature = feature
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x

def make_feature(cfg):
    layer = []
    in_channel = 3
    for v in cfg:
        if v == 'M':
            layer.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv = nn.Conv2d(in_channel, v, kernel_size=3, stride=1, padding=1)
            relu = nn.ReLU(inplace=True)
            layer.append(conv)
            layer.append(relu)
            in_channel = v

    return nn.Sequential(*layer)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(num_classes=1000):
    cfg = cfgs['vgg16']
    feature = make_feature(cfg)
    model = VGG(feature, num_classes)

    return model

def vgg11(num_classes=1000):
    cfg = cfgs['vgg11']
    feature = make_feature(cfg)
    model = VGG(feature, num_classes)

    return model

def vgg13(num_classes=1000):
    cfg = cfgs['vgg13']
    feature = make_feature(cfg)
    model = VGG(feature, num_classes)

    return model

def vgg19(num_classes=1000):
    cfg = cfgs['vgg19']
    feature = make_feature(cfg)
    model = VGG(feature, num_classes)

    return model



