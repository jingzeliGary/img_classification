'''
ResNet:
输入；[3,224,224]
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, **kwargs):
        super(BasicBlock, self).__init__()
        # 主路
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel * self.expansion)
        # 支路
        self.downsample = nn.Sequential()
        if stride != 1 or in_channel != out_channel * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * self.expansion)
            )

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = F.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1):
        super(Bottleneck, self).__init__()
        # 主路
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        # 支路
        self.downsample = nn.Sequential()
        if stride != 1 or in_channel != out_channel * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel * self.expansion)
            )

    def forward(self, x):
        identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = F.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, block_structure, class_nums):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_channel = 64
        self.layer1 = self.make_layer(block, block_structure[0], 64, stride=1)
        self.layer2 = self.make_layer(block, block_structure[1], 128, stride=2)
        self.layer3 = self.make_layer(block, block_structure[2], 256, stride=2)
        self.layer4 = self.make_layer(block, block_structure[3], 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输出 （1，1）
        self.fc = nn.Linear(512 * block.expansion, class_nums)

        # 初始化参数 kaiming_normal_
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


        '''
        # Xavier 均匀分布
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

        # Xavier 正态分布
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)

        # kaiming 均匀分布
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
        # model: fan_in 正向传播，方差一致; fan_out 反向传播, 方差一致

        # 正态分布
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=1)

        # 常量 , 一般是给网络中bias进行初始化
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.constant_(m.bias, val=0)

        '''

    def make_layer(self, block, block_num, conv_num, stride):
        strides = [stride] + [1] * (block_num - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, conv_num, stride))
            self.in_channel = conv_num * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)  # [64, 512*block.expension, 1, 1]
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(class_num=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], class_num=class_num)

def resnet34(class_num=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], class_nums=class_num)

def resnet50(class_num=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], class_num=class_num)

def resnet101(class_num=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], class_num=class_num)

def resnet152(class_num=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], class_num=class_num)

