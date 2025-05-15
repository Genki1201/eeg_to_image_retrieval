import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from tqdm import tqdm
from natsort import natsorted
import os
import numpy as np
import cv2
import pickle
import random
import gc

class EEGDataset(Dataset):
    def __init__(self, split_dic_path, data_path, label_path, split, network_name='EEGNet'):
        split_dic = torch.load(split_dic_path, map_location='cpu')
        data_dic = torch.load(data_path, map_location='cpu')
        self.dataset_lst = data_dic['dataset']
        self.means = data_dic.get("means", None)
        self.stddevs = data_dic.get("stddevs", None)
        del data_dic
        gc.collect()
        self.idx_lst = split_dic['splits'][0][split]
        self.network_name = network_name
  
        with open(label_path, 'rb') as f:
            self.label_dic = pickle.load(f)

    def __getitem__(self, i):
        idx = self.idx_lst[i]
        image_name = self.dataset_lst[idx]['image']
        image_embed = self.label_dic[image_name]
        eeg = self.dataset_lst[idx]['eeg'].float()

        # 正規化（平均・標準偏差が与えられていれば）
        if self.means is not None and self.stddevs is not None:
            # eeg = ((eeg - self.means) / self.stddevs).t()  # (T, C)　EEGNet
            eeg = ((eeg - self.means) / self.stddevs) # atms
        else:
            if self.network_name == 'EEGNet':
                eeg = eeg.t()
            else:
                eeg = eeg
        
        return eeg, image_embed

    def __len__(self):
        return len(self.idx_lst)

class EEG_small_dataset(Dataset):
    def __init__(self, eeg, file_names, split):
        self.eegs = eeg
        self.file_names = file_names
        self.split = split

        self.norm_max     = torch.max(self.eegs)
        self.norm_min     = torch.min(self.eegs)

        # labelファイルをロードしておく
        if split == 'train':
            # trainファイルから辞書をロード
            self.label_path = '/workspace/eeg_to_image_rcg/eeg_encoder/label_dic/small_dataset_feat_train.pkl'
        elif split == 'val':
            self.label_path = '/workspace/eeg_to_image_rcg/eeg_encoder/label_dic/small_dataset_feat_test.pkl'

        # pklファイルからロードしてくる
        with open(self.label_path, 'rb') as f:
            self.label_dic = pickle.load(f)

    def __getitem__(self, index):
        file_name = self.file_names[index]
        #  辞書からfile_nameをkeyとして
        image_feat = self.label_dic[file_name]

        eeg    = self.eegs[index]
        norm   = torch.max(eeg) / 2.0
        eeg    = (eeg - norm)/ norm
        # eeg    = (eeg - np.min(eeg))/ (np.max(eeg) - np.min(eeg))

        return eeg.T, image_feat

    def normalize_data(self, data):
        return (( data - self.norm_min ) / (self.norm_max - self.norm_min))

    def __len__(self):
        if len(self.eegs) == len(self.file_names):
            return len(self.eegs)


if __name__=='__main__':
    label_dir = './label_dic'
    split_dic_path = '/workspace/CVPR2021-02785/preprocessed/imagenet40-1000-1_split.pth'
    data_path = '/workspace/CVPR2021-02785/preprocessed/imagenet40-1000-1.pth'
    train_label_path = os.path.join(label_dir, 'train_imagefeat.pkl')
    val_label_path = os.path.join(label_dir, 'val_imagefeat.pkl')

    dataset = EEGDataset(split_dic_path, data_path, train_label_path, split='train')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=False, drop_last=True)

    val_dataset = EEGDataset(split_dic_path, data_path, val_label_path, split='val')
    val_dataloader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=False, drop_last=True)
    eeg, image_embed = val_dataset[0]
    print(type(eeg)) # tensor
    print(eeg.shape) # torch.Size([96, 512])
    print(type(image_embed)) # tensor
    print(image_embed.shape) # torch.Size([256])
    