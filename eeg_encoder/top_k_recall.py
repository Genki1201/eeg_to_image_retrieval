import os

from tqdm import tqdm
import numpy as np
import pdb
from natsort import natsorted
import cv2
from glob import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
import gc
import pickle
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from eeg_dataset import EEGDataset
from network import EEG_transformer_encoder, EEG_LSTM_original_model, classifier_EEGNet
from torch.nn.functional import cosine_similarity
from atms_network.network import ATMS

from torch.optim.lr_scheduler import LambdaLR

np.random.seed(45)
torch.manual_seed(45)


def top_k_recall(k=1, eeg_ckpt_path=None, image_encoder=None, network=None):
    if image_encoder=='clip':
        projection_dim = 1024 # moco 256, clip 1024
    elif image_encoder=='moco':
        projection_dim = 256
    elif image_encoder=='clip_in1k':
        projection_dim = 1280
    # eeg_encoder = ATMS().cuda()
    channel=96
    length=512
    if network == 'EEGNet':
        eeg_encoder = classifier_EEGNet(channel, length, projection_dim).cuda()
    elif network == 'ATMS':
        eeg_encoder = ATMS().cuda()
    # eeg_encoder = EEG_transformer_encoder(in_channels=96, in_timestep=512, hidden_size=int(projection_dim), projection_dim=projection_dim, num_layers=1, nhead=4, dropout=0).cuda()
    # eeg_encoder = EEG_LSTM_original_model().cuda()
    eegcheckpoint = torch.load(eeg_ckpt_path, map_location=torch.device("cuda"))
    eeg_encoder.load_state_dict(eegcheckpoint['model_state_dict'])

    # 検索元のEEG
    split_dic_path = '/workspace/CVPR2021-02785/preprocessed/imagenet40-1000-1_split.pth'
    data_path = '/workspace/CVPR2021-02785/preprocessed/imagenet40-1000-1.pth'
    split_dic = torch.load(split_dic_path, map_location='cpu')
    my_dic = torch.load(data_path, map_location='cpu')
    dataset_lst = my_dic['dataset']
    del my_dic
    gc.collect()
    idx_lst = split_dic['splits'][0]['test']

    # 検索対象の画像特徴量
    label_dir = './label_dic'
    if image_encoder=='clip':
        test_imagefeat_path = os.path.join(label_dir, 'test_clip_imagefeat.pkl')
    elif image_encoder=='moco':
        test_imagefeat_path = os.path.join(label_dir, 'test_imagefeat.pkl')
    elif image_encoder=='clip_in1k':
        test_imagefeat_path = os.path.join(label_dir, 'test_clip_in1k_imagefeat.pkl')
        
    with open(test_imagefeat_path, 'rb') as f:
        imagefeat_dic = pickle.load(f)
    imagefeat_keys = list(imagefeat_dic.keys())
    imagefeat_matrix = torch.stack([torch.tensor(imagefeat_dic[k]).cuda() for k in imagefeat_keys])  # shape: (N, 256)

    top_k_recall_lst = []
    correct_image_name = {}
    eeg_encoder.eval() 
    for idx in tqdm(idx_lst):
        image_name = dataset_lst[idx]['image']
        if network == 'EEGNet':
            eeg = dataset_lst[idx]['eeg'].T.cuda() # seq_len, channel
        else:
            eeg = dataset_lst[idx]['eeg'].cuda() # channel, seq_len

        with torch.no_grad():
            eeg_embed = eeg_encoder(eeg.unsqueeze(0))
        # コサイン類似度を計算
        cosine_similarities = cosine_similarity(eeg_embed, imagefeat_matrix, dim=1)  # shape: (N,)
        sorted_indices = torch.argsort(cosine_similarities, descending=True)  # 類似度が高い順のインデックス
        sorted_image_name = [imagefeat_keys[i] for i in sorted_indices]
        
        if image_name in sorted_image_name[:k]:
            top_k_recall_lst.append(1)
            correct_image_name[image_name] = sorted_image_name[:k]
        else:
            top_k_recall_lst.append(0)

    chance_level = math.comb(len(idx_lst)-1, k-1) / math.comb(len(idx_lst), k)
    print('chance level is ', chance_level)
    print(f'top {k} recall is ', sum(top_k_recall_lst)/len(top_k_recall_lst)) 
    with open(f"correct_image_name_k{k}_{image_encoder}.json", "w", encoding="utf-8") as f:
        json.dump(correct_image_name, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    image_encoder = 'clip_in1k' # 'clip' or 'moco' or clip_in1k
    network = 'EEGNet' # 'EEGNet'
    # top k recall　EEGを画像特徴量に変換して画像特徴量を検索する
    if image_encoder=='moco':
        eeg_ckpt_path = '/workspace/eeg_to_image_rcg/eeg_encoder/trained_eeg_encoder/EXPERIMENT_7_EEGNet@moco/bestckpt/eegfeat_4.80294527546052.pth' # best
        # eeg_ckpt_path = '/workspace/eeg_to_image_rcg/eeg_encoder/trained_eeg_encoder/EXPERIMENT_8_ATMS/bestckpt/eegfeat_4.172689583993727.pth'
    elif image_encoder=='clip':
        eeg_ckpt_path = '/workspace/eeg_to_image_rcg/eeg_encoder/trained_eeg_encoder/EXPERIMENT_11_EEGNet@clip/bestckpt/eegfeat_4.845380337007584.pth'
    elif image_encoder=='clip_in1k':
        eeg_ckpt_path = '/workspace/eeg_to_image_rcg/eeg_encoder/trained_eeg_encoder/EXPERIMENT_12_EEGNet@clip_in1k/bestckpt/eegfeat_4.864214712573636.pth'
    top_k_recall(k=5, eeg_ckpt_path=eeg_ckpt_path, image_encoder=image_encoder, network=network)
