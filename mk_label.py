# EEGエンコーダを訓練させるときの正解ラベルのリストを作り保存する

import math
import sys
from typing import Iterable
import os
import torch
# import util.misc as misc
import util.lr_sched as lr_sched
import cv2
# import torch_fidelity
import numpy as np
import shutil
from PIL import Image
import clip
import open_clip

from tqdm import tqdm
from eeg_encoder.image_dataset import Image_dataset, ImageNet_dataset
import pickle


def gen_big_img_feat(model, args, split_dic_path, data_path, stimuli_path, split): # MoCo v3の画像特徴量を取得
    split_dic = torch.load(split_dic_path, map_location='cpu')
    my_dic = torch.load(data_path, map_location='cpu')
    dataset_lst = my_dic['dataset']
    del my_dic
    idx_lst = split_dic['splits'][0][split] # 5パターンある cross validation用
    eeg_encoder_label_dic = {}

    model.eval()
    print('extracting image feature...')
    for idx in tqdm(idx_lst):
        image_name = dataset_lst[idx]['image']
        image_path = os.path.join(stimuli_path, image_name + '.JPEG')
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
        image_tensor /= 255.0
        image = image_tensor.cuda()

        with torch.no_grad():
            sampled_rep = model(image, None,
                                gen_image=True, bsz=1,
                                choice_temperature=args.temp,
                                sampled_rep=None,
                                rdm_steps=args.rdm_steps, eta=args.eta, cfg=0, return_rep=True)
        sampled_rep = sampled_rep.squeeze()
        sampled_rep = sampled_rep.detach().cpu()

        eeg_encoder_label_dic[image_name] = sampled_rep

    save_path = 'eeg_encoder/label_dic/{}_imagefeat.pkl'.format(split) # train or test
    with open(save_path, 'wb') as f:
        pickle.dump(eeg_encoder_label_dic, f)

    return eeg_encoder_label_dic

def gen_small_img_feat(model, args, split):
    # datasetをインスタンス化してdataloaderを定義（バッチサイズ1）
    cvpr40_train_path = '/workspace/simple_rcg/eeg_encoder/eeg_imagenet40_cvpr_2017_raw/'+split+'/' # train or test
    dataset = Image_dataset(cvpr40_train_path)
    train_dl = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=args.pin_mem)
    eeg_encoder_label_dic = {}

    model.eval()
    print('generatirn image feature...')
    for file_name, image in tqdm(train_dl):
        image = image.cuda()
        with torch.no_grad():
            sampled_rep = model(image, None,
                                gen_image=True, bsz=1,
                                choice_temperature=args.temp,
                                sampled_rep=None,
                                rdm_steps=args.rdm_steps, eta=args.eta, cfg=0, return_rep=True)
        sampled_rep = sampled_rep.squeeze()
        sampled_rep = sampled_rep.detach().cpu()

        eeg_encoder_label_dic[file_name[0]] = sampled_rep

    save_path = 'eeg_encoder/label_dic/small_dataset_feat_{}.pkl'.format(split) # train or test
    with open(save_path, 'wb') as f:
        pickle.dump(eeg_encoder_label_dic, f)

    return eeg_encoder_label_dic

def gen_imagenet_feat(model, args):
    # datasetをインスタンス化してdataloaderを定義（バッチサイズ1）
    imagenet_train_path = '/workspace/simple_rcg/imagenet_train_all/'
    dataset = ImageNet_dataset(imagenet_train_path)
    train_dl = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=args.pin_mem)
    eeg_encoder_label_dic = []

    model.eval()
    print('generatirn image feature...')
    for image in tqdm(train_dl):
        image = image.cuda()
        with torch.no_grad():
            sampled_rep = model(image, None,
                                gen_image=True, bsz=1,
                                choice_temperature=args.temp,
                                sampled_rep=None,
                                rdm_steps=args.rdm_steps, eta=args.eta, cfg=0, return_rep=True)
        sampled_rep = sampled_rep.squeeze()
        sampled_rep = sampled_rep.detach().cpu()

        eeg_encoder_label_dic.append(sampled_rep)

    save_path = 'eeg_encoder/label_dic/imagenet_feature.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(eeg_encoder_label_dic, f)

    return eeg_encoder_label_dic


def gen_big_clip_img_feat(split_dic_path, data_path, stimuli_path, split): # CLIPの画像特徴量を取得
    split_dic = torch.load(split_dic_path, map_location='cpu')
    my_dic = torch.load(data_path, map_location='cpu')
    dataset_lst = my_dic['dataset']
    del my_dic
    idx_lst = split_dic['splits'][0][split] # 5パターンある cross validation用
    eeg_encoder_label_dic = {}

    model, preprocess, tokenizer = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
    image_encoder = model.visual
    image_encoder.eval().cuda() 
    print('extracting image feature...')
    for idx in tqdm(idx_lst):
        image_name = dataset_lst[idx]['image']
        image_path = os.path.join(stimuli_path, image_name + '.JPEG')
        image = preprocess(Image.open(image_path)).unsqueeze(0).cuda() 

        with torch.no_grad():
            sampled_rep = image_encoder(image)
        sampled_rep = sampled_rep.squeeze()
        sampled_rep = sampled_rep.detach().cpu()

        eeg_encoder_label_dic[image_name] = sampled_rep

    save_path = 'eeg_encoder/label_dic/{}_clip_imagefeat.pkl'.format(split) # train or test
    with open(save_path, 'wb') as f:
        pickle.dump(eeg_encoder_label_dic, f)

    return eeg_encoder_label_dic

if __name__ == '__main__':
    split_dic_path = '/workspace/CVPR2021-02785/preprocessed/imagenet40-1000-1_split.pth'
    data_path = '/workspace/CVPR2021-02785/preprocessed/imagenet40-1000-1.pth'
    stimuli_path = '/workspace/CVPR2021-02785/stimuli'
    split_lst = ['train', 'val', 'test']
    for split in split_lst:
        gen_big_clip_img_feat(split_dic_path, data_path, stimuli_path, split)
