'''
打乱， 划分数据和对应的标签， 保存 .npy
'''

import os
import random
import numpy as np
import tqdm
import cv2


root = './animals10/raw-img'
class_list = os.listdir(root)
print(class_list)

image_list = []
label_list = []

for index, class_name in enumerate(class_list):
    img_list = os.listdir(os.path.join(root, class_name))
    for img_name in tqdm.tqdm(img_list):
        img = cv2.imread( os.path.join(root, class_name, img_name))

        image_list.append(img)  # List(numpy(h,w,c))
        label_list.append(index)  # List(int)

# 打乱
X = []
y = []

index = [i for i in range(len(image_list))]
random.shuffle(index)
for i in index:
    img = image_list[i]
    label = label_list[i]

    X.append(img)
    y.append(label)

# 划分
train_ratio = 0.7
train_img = X[:int(len(X) * train_ratio)]
train_label = y[:int(len(y) * train_ratio)]
test_img = X[int(len(X) * train_ratio):]
test_label = y[int(len(y) * train_ratio):]

# 保存
train_img = np.asarray(train_img)
train_label = np.asarray(train_label)
test_img = np.asarray(test_img)
test_label = np.asarray(test_label)
label_list = np.asarray(class_list)

np.save('../data/train_img.npy', train_img)
np.save('../data/train_label.npy', train_label)
np.save('../data/test_img.npy', test_img)
np.save('../data/test_label.npy', test_label)
np.save('../data/label_list.npy', label_list)
