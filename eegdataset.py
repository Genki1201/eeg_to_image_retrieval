import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from natsort import natsorted
import os
import numpy as np
import cv2


class EEG_to_Image_dataset(Dataset):
    def __init__(self, dataset_path, image_class):
        self.dataset_path = dataset_path
        self.image_class = image_class
        self.file_name_lst = []
        self.image_name_lst = []
        self.push_list(dataset_path + 'test')
        # self.push_list(dataset_path + 'val')

    def push_list(self, dataset_path):
        for file_name in natsorted(os.listdir(dataset_path)):
            if file_name.split('_')[0]==str(self.image_class): #and file_name.split('_')[1] not in self.image_name_lst:
                self.file_name_lst.append(file_name)
                # print(file_name.split('_')[1])
                self.image_name_lst.append(file_name.split('_')[1])


    def __getitem__(self, index):
        file_name = self.file_name_lst[index]

        if os.path.exists(self.dataset_path + 'test/' + file_name):
            loaded_array = np.load(self.dataset_path + 'test/' + file_name, allow_pickle=True)
        else:
            loaded_array = np.load(self.dataset_path + 'val/' + file_name, allow_pickle=True)
        #image
        img = cv2.resize(loaded_array[0], (224, 224))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).to(torch.float32)
        # eeg
        eeg = loaded_array[1].T
        eeg = torch.from_numpy(eeg).to(torch.float32)
        norm   = torch.max(eeg) / 2.0
        eeg    = (eeg - norm)/ norm

        return img, eeg

    def __len__(self):
        return len(self.file_name_lst)

def check_dataset(dataset_path):
    test_path = dataset_path+'/test/'
    class_lst = []

    for file_name in natsorted(os.listdir(test_path)):
        loaded_array = np.load(test_path + file_name, allow_pickle=True)
        img = cv2.resize(loaded_array[0], (224, 224))
        test_eeg = np.load(test_path + file_name, allow_pickle=True)[1]

        if (validation_eeg != test_eeg).any():
            print('different')
        if file_name.split('_')[0] not in class_lst:
            class_lst.append(file_name.split('_')[0])
    print(class_lst)
    

if __name__=='__main__':
    check_dataset('/workspace/simple_rcg/eeg_encoder/eeg_imagenet40_cvpr_2017_raw') # validation dataはすべてvalidationに漏れ出している