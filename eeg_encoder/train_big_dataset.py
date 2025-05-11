import os

from tqdm import tqdm
import numpy as np
import pdb
from natsort import natsorted
import cv2
from glob import glob
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from eeg_dataset import EEGDataset
from network import EEG_transformer_encoder, EEG_LSTM_original_model

from torch.optim.lr_scheduler import LambdaLR

# np.random.seed(45)
# torch.manual_seed(45)


def train(epoch, eeg_encoder, optimizer, train_dataloader):
    eeg_encoder_loss = []
    temperature = 0.5
    tq = tqdm(train_dataloader)
    for batch_idx, (eeg, image_embed) in enumerate(tq, start=1):
        eeg = eeg.cuda()
        image_embed = image_embed.cuda()

        optimizer.zero_grad()

        eeg_embed = eeg_encoder(eeg)
        logits = (eeg_embed @ image_embed.T) * torch.exp(torch.tensor(temperature))
        # logits = (eeg_embed @ image_embed.T) / temperature # temp 0.07

        labels = torch.arange(image_embed.shape[0]).cuda()
        
        loss_i = F.cross_entropy(logits, labels, reduction='none')
        loss_t = F.cross_entropy(logits.T, labels, reduction='none')

        loss = (loss_i + loss_t) / 2.0
        loss = loss.mean()

        loss.backward()
        optimizer.step()

        eeg_encoder_loss.append(loss.item())

    avg_loss_encoder = sum(eeg_encoder_loss) / len(eeg_encoder_loss)

    return avg_loss_encoder

def validation(epoch, eeg_encoder, optimizer, val_dataloader):
    eeg_encoder_loss = []
    temperature = 0.5
    for batch_idx, (eeg, image_embed) in enumerate(val_dataloader, start=1):
        eeg, image_embed = eeg.cuda(), image_embed.cuda()
        with torch.no_grad():
            eeg_embed = eeg_encoder(eeg)
            logits = (eeg_embed @ image_embed.T) * torch.exp(torch.tensor(temperature))
            # logits = (eeg_embed @ image_embed.T) / temperature # temp 0.07

            labels = torch.arange(image_embed.shape[0]).cuda()
            
            loss_i = F.cross_entropy(logits, labels, reduction='none')
            loss_t = F.cross_entropy(logits.T, labels, reduction='none')

            loss = (loss_i + loss_t) / 2.0
            loss = loss.mean()

            eeg_encoder_loss.append(loss.detach().cpu().numpy())

    return sum(eeg_encoder_loss) / len(eeg_encoder_loss)

def eeg_encoder_train():

    label_dir = './label_dic'
    split_dic_path = '/workspace/CVPR2021-02785/preprocessed/imagenet40-1000-1_split.pth'
    data_path = '/workspace/CVPR2021-02785/preprocessed/imagenet40-1000-1.pth'
    model_type = 'transformer' # lstm or transformer
    image_encoder = 'moco' # 'clip' or 'moco'

    # ## hyperparameters
    batch_size     = 128 # default 128
    EPOCHS         = 3000

    class_labels   = {}
    label_count    = 0

    print('loading train dataset...')
    if image_encoder == 'moco':
        train_label_path = os.path.join(label_dir, 'train_imagefeat.pkl')
    elif image_encoder == 'clip':
        train_label_path = os.path.join(label_dir, 'train_clip_imagefeat.pkl')
    train_dataset = EEGDataset(split_dic_path, data_path, train_label_path, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True)

    ## Validation data
    print('loading val dataset...')
    if image_encoder == 'moco':
        train_label_path = os.path.join(label_dir, 'val_imagefeat.pkl')
    elif image_encoder == 'clip':
        train_label_path = os.path.join(label_dir, 'val_clip_imagefeat.pkl')
    val_dataset = EEGDataset(split_dic_path, data_path, val_label_path, split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=True)

    ## eeg_encoder
    if model_type=='transformer':
        if image_encoder == 'moco':
            projection_dim = 256
        elif image_encoder == 'clip':
            projection_dim = 1024 # moco 256, clip 1024
        eeg_encoder = EEG_transformer_encoder(in_channels=96, in_timestep=512, hidden_size=int(projection_dim / 2), projection_dim=projection_dim, num_layers=1, nhead=4, dropout=0).cuda() # num_layers=1
    else:
        eeg_encoder = EEG_LSTM_original_model().cuda()

    print(eeg_encoder)

    optimizer = torch.optim.Adam(\
                                    list(eeg_encoder.parameters()),\
                                    lr=3e-4,\
                                    betas=(0.9, 0.999)
                                )
    
    # lambda_lr = lambda epoch: 0.999 ** epoch
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)
    
    ## save directory
    dir_info  = natsorted(glob('trained_eeg_encoder/EXPERIMENT_*'))
    if len(dir_info)==0:
        experiment_num = 1
    else:
        experiment_num = int(dir_info[-1].split('_')[-2]) + 1


    experiment_path = 'trained_eeg_encoder/EXPERIMENT_{}_{}'.format(experiment_num, model_type)
    
    if not os.path.isdir(experiment_path):
        os.makedirs(experiment_path)

    START_EPOCH = 0

    os.makedirs(experiment_path+'/checkpoints/')
    os.makedirs(experiment_path+'/bestckpt/')
    os.makedirs(experiment_path+'/best_train_ckpt/')
    os.makedirs(experiment_path+'/per500_ckpts/')

    best_val_loss   = 1000000
    best_val_epoch = 0

    best_train_loss = 1000000
    best_trian_epoch = 0

    train_loss_lst = []
    val_loss_lst = []

    for epoch in range(START_EPOCH, EPOCHS):

        print('start training epoch ', epoch)
        encoder_train_loss = train(epoch, eeg_encoder, optimizer, train_dataloader)
        train_loss_lst.append(encoder_train_loss)

        encoder_val_loss = validation(epoch, eeg_encoder, optimizer, val_dataloader)
        # scheduler.step()
        val_loss_lst.append(encoder_val_loss)
        print('training loss:', encoder_train_loss, ', validation loss:', encoder_val_loss)

        if encoder_val_loss < best_val_loss:
            files = glob(experiment_path + '/bestckpt/*')
            if files:
                for file in files:
                    os.remove(file)

            best_val_loss = encoder_val_loss
            best_val_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': eeg_encoder.state_dict(),
            }, experiment_path + '/bestckpt/eegfeat_{}.pth'.format(encoder_val_loss))

        if encoder_train_loss < best_train_loss:
            files = glob(experiment_path + '/best_train_ckpt/*')
            if files:
                for file in files:
                    os.remove(file)

            best_train_loss = encoder_train_loss
            best_train_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': eeg_encoder.state_dict(),
            }, experiment_path + '/best_train_ckpt/eegfeat_{}.pth'.format(encoder_train_loss))


        torch.save({
                'epoch': epoch,
                'model_state_dict': eeg_encoder.state_dict(),
              }, experiment_path+'/checkpoints/eegfeat.pth')

        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': eeg_encoder.state_dict(),
            }, experiment_path + '/per500_ckpts/epoch_{}_{}.pth'.format(epoch, encoder_train_loss))
        
        if epoch % 10 == 0:
            # トレーニング損失のグラフを作成
            write_loss_graph(lst=train_loss_lst, label='train_loss', save_path=experiment_path+'/train_loss.png', title=f'Train Loss')

            # バリデーション損失のグラフを作成
            write_loss_graph(lst=val_loss_lst, label='val_loss', save_path=experiment_path+'/val_loss.png', title=f'Val Loss')

            write_loss_graph(lst=train_loss_lst, label='train_loss', lst_sub=val_loss_lst, label_sub='val_loss',
                            save_path=experiment_path+'/combined_loss.png', title=f'Train and Validation Loss')


def write_loss_graph(lst, label, lst_sub=None, label_sub=None, save_path=None, title=None):
    save_path = save_path
    plt.figure()
    plt.plot(lst, label=label)
    if lst_sub is not None:
        plt.plot(lst_sub, label=label_sub)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)

    plt.savefig(save_path)
    plt.close()


if __name__ == '__main__':
    eeg_encoder_train()
