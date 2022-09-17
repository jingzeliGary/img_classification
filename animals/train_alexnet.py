'''
训练模型：
1. 预处理
2. 数据加载
3. 加载模型
4. 训练, 评估， 保存模型

# AlexNet, epochs=20, lr=0.001, best_acc= 0.6864174620007069
'''
import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model.alexnet import AlexNet
from dataset.mydataset import MyDataset
from train_utils.train_test_one_epoch import train_epoch, test_epoch
import matplotlib.pyplot as plt
from train_utils.train_test_one_epoch import create_lr_scheduler

# 预处理
train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

# 加载train data , test data
train_img = np.load('./data/train_img.npy', allow_pickle=True)
train_label = np.load('./data/train_label.npy', allow_pickle=True)
test_img = np.load('./data/test_img.npy', allow_pickle=True)
test_label = np.load('./data/test_label.npy', allow_pickle=True)
lable_list = np.load('./data/label_list.npy', allow_pickle=True)
print('标签:', lable_list)
print('训练集图片的数量:', len(train_img))
print('训练集标签的数量:', len(train_label))
print('测试集图片的数量:', len(test_img))
print('测试集标签的数量:', len(test_label))

train_set = MyDataset(train_img, train_label, transform = train_transforms)
test_set = MyDataset(test_img, test_label, transform = test_transforms)


# 数据加载
batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=train_set.collate_fn)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=test_set.collate_fn)

# 加载模型
model = AlexNet(num_classes=10)

# 结果保存地址
if not os.path.exists('results'):
    os.makedirs('results')

results_file = "./results/results_AlexNet.txt"

# 训练
epochs = 20
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Use {} to training'.format(device))
model.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs, warmup=True)

# 保存最优权重地址
save_path = './best_weights/AlexNet.pth'
best_acc = 0.0
loss_train = []
loss_test = []

for epoch in range(epochs):
    print('This is epoch: %s ' %epoch)
    # train
    train_loss, lr = train_epoch(train_loader, model, device, optimizer, loss_function, epoch, lr_scheduler, print_freq=10)

    # test
    test_loss, acc = test_epoch(test_loader, model, device, loss_function)

    # 写入结果
    with open(results_file, "a") as f:
        # 记录每个epoch对应的train_loss、lr以及验证集各指标
        train_info = f"[epoch: {epoch}]\n" \
                     f"train_loss: {train_loss:.4f}\n" \
                     f"lr: {lr:.6f}\n"
        test_info = f"test_loss: {test_loss}\n" \
                     f"acc: {acc:.4f}\n"
        f.write(train_info + test_info + "\n\n")

    # 保存最优acc 权重
    if best_acc < acc:
        best_acc = acc
        torch.save(model.state_dict(), save_path)

    loss_train.append(train_loss)
    loss_test.append(test_loss)

print('The best acc is %s' %best_acc)
print('Training Finished')

# 可视化 plt--- cpu-numpy
loss_train = np.array(torch.tensor(loss_train, device='cpu'))
loss_test = np.array(torch.tensor(loss_test, device='cpu'))

plt.figure(figsize=(10,5))
plt.title('Train and Test Loss')
plt.plot(loss_train, label='train loss')
plt.plot(loss_test, label='test loss')
plt.xlabel('nums_epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./train_test_loss_for_AlexNet.png')