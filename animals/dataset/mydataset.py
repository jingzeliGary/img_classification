'''
自定义数据集
'''

from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, img_list, label_list, transform=None):
        self.img_list = img_list
        self.label_list = label_list
        self.transform = transform

    def __getitem__(self, item):
        image = self.img_list[item]
        label = self.label_list[item]
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.img_list)


