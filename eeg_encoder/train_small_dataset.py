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
from eeg_dataset import EEG_small_dataset
from network import EEG_transformer_encoder, EEG_LSTM_original_model, EEG_LSTM

from cosine_loss import CosineSimilarityLoss
from torch.optim.lr_scheduler import LambdaLR

np.random.seed(45)
torch.manual_seed(45)


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

def eeg_encoder_train(unknown_class=None, unknown_class_name=None):

    base_path       = '/workspace/simple_rcg/eeg_encoder/eeg_imagenet40_cvpr_2017_raw/'
    train_path      = 'train/'
    # validation_path = 'val/'
    validation_path = 'test/'
    only_head = False # 後ろのみでできる

    #load the data
    ## Training data
    x_train_eeg = []
    train_file_names = []

    # ## hyperparameters
    batch_size     = 64 # default 128
    EPOCHS         = 3000

    class_labels   = {}
    label_count    = 0

    print('loading train dataset...')
    for file_name in tqdm(natsorted(os.listdir(base_path + train_path))):
        if unknown_class is None or file_name.split('_')[0] != unknown_class:
            loaded_array = np.load(base_path + train_path + file_name, allow_pickle=True)
            eeg = loaded_array[1]
            x_train_eeg.append(eeg.T)
            train_file_names.append(file_name)

    if unknown_class is not None:
        print('unknown class :', unknown_class_name)
        print('training data size: ', len(train_file_names))

    x_train_eeg   = np.array(x_train_eeg)
    x_train_eeg   = torch.from_numpy(x_train_eeg).float().cuda()

    train_data       = EEG_small_dataset(x_train_eeg, train_file_names, split='train')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True)


    ## Validation data
    x_val_eeg = []
    val_file_names = []

    print('loading test dataset...')
    for file_name in tqdm(natsorted(os.listdir(base_path + validation_path))):
        loaded_array = np.load(base_path + validation_path + file_name, allow_pickle=True)
        eeg = loaded_array[1]
        x_val_eeg.append(eeg.T)
        val_file_names.append(file_name)

    x_val_eeg   = np.array(x_val_eeg)
    x_val_eeg   = torch.from_numpy(x_val_eeg).float().cuda()

    val_data       = EEG_small_dataset(x_val_eeg, val_file_names, split='val')
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=True)

    ## eeg_encoder
    model_type = 'lstm'
    if model_type == 'transformer':
        eeg_encoder = EEG_transformer_encoder(in_channels=128, in_timestep=440, hidden_size=128, projection_dim=256, num_layers=1, nhead=4, dropout=0).cuda() # num_layers=1
    elif model_type == 'lstm':
        eeg_encoder = EEG_LSTM().cuda()

    print(eeg_encoder)

    optimizer = torch.optim.Adam(\
                                    list(eeg_encoder.parameters()),\
                                    lr=3e-4,\
                                    betas=(0.9, 0.999)
                                )

    # lambda_lr = lambda epoch: 0.999 ** epoch
    # scheduler = LambdaLR(optimizer, lr_lambda=lambda_lr)

    ## save directory
    dir_info  = natsorted(glob('trained_small_eeg_encoder/EXPERIMENT_*'))
    if len(dir_info)==0:
        experiment_num = 1
    else:
        experiment_num = int(dir_info[-1].split('_')[-2]) + 1

    if unknown_class_name is not None:
        experiment_path = 'trained_small_eeg_encoder/EXPERIMENT_{}_{}_{}_{}'.format(experiment_num, loss_type, model_type, unknown_class_name)
    else:
        experiment_path = 'trained_small_eeg_encoder/EXPERIMENT_{}_{}'.format(experiment_num, model_type)

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
            write_loss_graph(lst=train_loss_lst, label='train_loss', save_path=experiment_path+'/train_loss.png', title=f'Train Loss {unknown_class_name}')

            # バリデーション損失のグラフを作成
            write_loss_graph(lst=val_loss_lst, label='val_loss', save_path=experiment_path+'/val_loss.png', title=f'Val Loss {unknown_class_name}')

            write_loss_graph(lst=train_loss_lst, label='train_loss', lst_sub=val_loss_lst, label_sub='val_loss',
                            save_path=experiment_path+'/combined_loss.png', title=f'Train and Validation Loss {unknown_class_name}')


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