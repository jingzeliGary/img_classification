'''
使用训练好的 resnet34, 进行预测
FPS： 435
 '''
import time

import torch
import torchvision.transforms as transforms
from model.alexnet import AlexNet
from model.resnet import resnet34
import cv2
import numpy as np
import matplotlib.pyplot as plt

test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

label_list = np.load('./data/label_list.npy')
print('标签', label_list)

# 读取预测图片
img = cv2.imread('./pred_img/sheep.png')
# print(img.shape)  # numpy (h,w,c)
# 预处理图片
img = test_transforms(img)
img = torch.unsqueeze(img, dim=0) # [1,C,H,W]

# 加载模型和最优权重
model = resnet34(class_num=10)
model.load_state_dict(torch.load('./best_weights/ResNet34.pth'))
# print(model)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model.to(device)
img = img.to(device)

# prediction
model.eval()
with torch.no_grad():
    evaluation_time = time.time()
    pred = model(img)  # pred [1,10]

    pred = torch.squeeze(pred)  # [10]
    pred = torch.softmax(pred, dim=0)  # dim=0 在第一维的 sum=1
    predict_class = torch.argmax(pred)
    class_name = label_list[predict_class]

    evaluation_time = time.time() - evaluation_time  # 处理一张图片所用时间(ms)

    print('预测类别索引', predict_class)
    print('预测类别', class_name)
    print('FPS:', (1.0 / evaluation_time) * 1000) # FPS： 一秒处理的图片数量




