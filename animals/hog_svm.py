"""
1. 读取数据
2. 提取HOG特征
3. 加载SVM模型
4. 训练
5. 评估
6. 保存模型
7. 可视化训练结果

acc = 0.40106951871657753
"""
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn import svm
import os

# 加载train data , test data
train_img = np.load('./data/train_img.npy', allow_pickle=True)
train_label = np.load('./data/train_label.npy', allow_pickle=True)
test_img = np.load('./data/test_img.npy', allow_pickle=True)
test_label = np.load('./data/test_label.npy', allow_pickle=True)
label_list = np.load('./data/label_list.npy', allow_pickle=True)
print('标签:', label_list)
print('训练集图片的数量:', len(train_img))
print('训练集标签的数量:', len(train_label))
print('测试集图片的数量:', len(test_img))
print('测试集标签的数量:', len(test_label))

train_feature_list = []
test_feature_list = []

# 提取 hog 特征
for image, label_index in zip(train_img, train_label):
    '''
    image: numpy(h,w,c)
    label_name: str
    '''
    image = cv2.resize(image, (100,100))
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    # hog 提取
    feature = hog(image=gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

    # 存入特征列表和标签列表
    train_feature_list.append(feature)


for image, label_name in zip(test_img, test_label):
    '''
    image: numpy(h,w,c)
    label_name: str
    '''
    image = cv2.resize(image, (100,100))
    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    # hog 提取
    feature = hog(image=gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

    # 存入特征列表和标签列表
    test_feature_list.append(feature)

# 加载SVM
cls = svm.SVC(kernel='linear')  # rbf, linear, poly

# 训练
cls.fit(train_feature_list, train_label)

# 评估
pred = cls.predict(test_feature_list)
print(pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(test_label, pred)
print(acc)

# 保存训练好的模型
from joblib import dump,load
if not os.path.exists('trained_model'):
    os.makedirs('./trained_model')

dump(cls,'./trained_model/svm_linear.joblib')

# 可视化融合矩阵
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

cm = confusion_matrix(test_label, pred)
df_cm = pd.DataFrame(cm, index=[i for i in ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']],
                     columns=[i for i in ['cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo']])
plt.figure(figsize=(10,7))
sn.heatmap(df_cm,annot=True, cmap='Greens',fmt='d')
plt.savefig('./hog_svm_heatmap.png')
