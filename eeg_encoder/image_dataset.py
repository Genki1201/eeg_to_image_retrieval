# Image Datasetクラスを定義 ファイル名のリストを作っておき、indexが来たらそのファイル名とimageを返す。file_name, image
# インスタンス変数はpath

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from natsort import natsorted
import os
import numpy as np
import cv2


class Image_dataset(Dataset):
    def __init__(self, dataset_path, mk_class=None):
        self.dataset_path = dataset_path
        self.mk_class = mk_class
        self.file_name_lst = []
        
        for file_name in natsorted(os.listdir(dataset_path)):
            if mk_class is not None:
                if file_name.split('_')[0]==str(self.mk_class.split('_')[0]): #and file_name.split('_')[1] not in self.image_name_lst:
                    self.file_name_lst.append(file_name)
            else:
                self.file_name_lst.append(file_name)


    def __getitem__(self, index):
        file_name = self.file_name_lst[index]

        loaded_array = np.load(self.dataset_path + file_name, allow_pickle=True)
        img = cv2.resize(loaded_array[0], (224, 224))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        return file_name, img

    def __len__(self):
        return len(self.file_name_lst)


class ImageNet_dataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.file_name_lst = []
        
        for class_id in natsorted(os.listdir(dataset_path)):
            class_dir = os.path.join(dataset_path, class_id) 
            if not os.path.isdir(class_dir):
                continue
            for file_name in natsorted(os.listdir(class_dir)):
                file_path = os.path.join(class_id, file_name)
                self.file_name_lst.append(file_path)

    def __getitem__(self, index):
        image_path = self.file_name_lst[index]
        image_path = os.path.join(self.dataset_path, image_path)

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
        image_tensor /= 255.0

        return image_tensor

    def __len__(self):
        return len(self.file_name_lst)
