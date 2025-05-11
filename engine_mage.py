import math
import sys
from typing import Iterable
import os
import torch
import util.misc as misc
import util.lr_sched as lr_sched
import cv2
import torch_fidelity
import numpy as np
import shutil
from PIL import Image
from tqdm import tqdm

from eegdataset import EEG_to_Image_dataset
# from eeg_encoder.network import EEG_Transformer_model, EEG_LSTM_model, CNN_model, EEG_LSTM_3_model, EEG_LSTM_original_model, EEG_BiLSTM_model, EEG_BiLSTM_original_model
from torchvision.transforms import ToPILImage
import pickle
from eeg_encoder.image_dataset import Image_dataset
import lpips
import torchvision.transforms as transforms

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

def gen_img(model, args, epoch, batch_size=16, log_writer=None, cfg=0.0, make_class=None, dataset_path=None): # goalimageのdataloaderが入ってくるようにする
    model.eval()
    num_steps = args.num_images // (batch_size * misc.get_world_size()) + 1
    save_folder = "CVPR40_Image2Image_train/" + make_class + '/'
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    dataset = Image_dataset(dataset_path, make_class)
    dl = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=args.pin_mem, shuffle=True)

    for file_name, img in dl: # ゴール画像のdataloader
        with torch.no_grad():
            gen_images_batch, _ = model(img, None, # 最初のNoneに画像を入れる必要がある。 goal_images_batch
                                        gen_image=True, bsz=batch_size,
                                        choice_temperature=args.temp,
                                        num_iter=args.num_iter, sampled_rep=None,
                                        rdm_steps=args.rdm_steps, eta=args.eta, cfg=cfg)
        gen_images_batch = misc.concat_all_gather(gen_images_batch) # 全てのGPUから結果を集約
        gen_images_batch = gen_images_batch.detach().cpu()

        # save img
        print('saveing image...')
        gen_img = np.clip(gen_images_batch[0].numpy().transpose([1, 2, 0]) * 255, 0, 255)
        gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(
            os.path.join(save_folder, '{}.png'.format(str(file_name).zfill(5))),
            gen_img)
        goal_img = np.clip(img[0].numpy().transpose([1, 2, 0]) * 255, 0, 255)
        goal_img = goal_img.astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(
                os.path.join(save_folder, '{}.png'.format(str(file_name).zfill(5) + '_goal')),
                goal_img)
        print(os.path.join(save_folder, '{}.png'.format(str(file_name).zfill(5) + '_goal')))
        print('completed!')


def eeg_to_image(image_generator, output_folder, args, batch_size=16, log_writer=None, cfg=0.0, make_class=None, make_class_name=None, eegckpt=None, model_type=None):
    # datasetをインスタンス化してdataloaderを定義（バッチサイズ1）
    cvpr40_test_path = 'eeg_encoder/eeg_imagenet40_cvpr_2017_raw/' # train or test
    print('loading dataset...')
    dataset = EEG_to_Image_dataset(cvpr40_test_path, make_class)
    dl = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=args.pin_mem, shuffle=True)
    save_folder = output_folder + make_class_name
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    if model_type=='transformer':
        eeg_encoder = EEG_Transformer_model(hidden_size=128, projection_dim=256, num_layers=4).cuda()
    elif model_type=='lstm':
        eeg_encoder = EEG_LSTM_model(n_features=128, projection_dim=256, num_layers=4).cuda()
    elif model_type=='cnn':
        eeg_encoder = CNN_model().cuda()

    if eegckpt is not None:
        eegcheckpoint = torch.load(eegckpt, map_location=torch.device("cuda"))
        eeg_encoder.load_state_dict(eegcheckpoint['model_state_dict'])
    else:
        print('There is no eeg chekpoint...')
    eeg_encoder.eval()
    image_generator.eval()
    to_pil = ToPILImage()

    print('generating image...')
    for idx, (image, eeg) in enumerate(dl):
        print('generating: ', idx+1, '/', len(dl))
        goal_images = image
        with torch.no_grad():
            eeg = eeg.cuda()
            rep = eeg_encoder(eeg)
            # 標準化
            rep_std = torch.std(rep, dim=1, keepdim=True)
            rep_mean = torch.mean(rep, dim=1, keepdim=True)
            rep = (rep - rep_mean) / rep_std # eeg特徴量

            eeg_rep = rep

            gen_images_batch, _ = image_generator(goal_images, None, # 最初のNoneに画像を入れる必要がある。 goal_images_batch
                                        gen_image=True, bsz=batch_size,
                                        choice_temperature=args.temp,
                                        num_iter=args.num_iter, sampled_rep=None,
                                        rdm_steps=args.rdm_steps, eta=args.eta, cfg=cfg,
                                        eeg2image=True, eeg_rep=eeg_rep)
        gen_images_batch = misc.concat_all_gather(gen_images_batch) # 全てのGPUから結果を集約
        gen_images_batch = gen_images_batch.detach().cpu()

        # save img
        goal_image = to_pil(goal_images[0])
        gen_image = to_pil(gen_images_batch[0])
        grid = image_grid([goal_image, gen_image.resize((224, 224))], 1, 2)

        output_path = save_folder+'/{}.png'.format(idx)
        print(output_path)
        grid.save(output_path, format='PNG')


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def cv2_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols, "Number of images must match rows * cols"
    
    h, w, _ = imgs[0].shape

    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    for i, img in enumerate(imgs):
        row = i // cols
        col = i % cols
        
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img

    return grid

def evaluate(image_generator, output_folder, args, batch_size=1, log_writer=None, cfg=0.0, make_class=None, make_class_name=None, eegckpt=None, model_type=None):
    # datasetをインスタンス化してdataloaderを定義（バッチサイズ1）
    cvpr40_test_path = 'eeg_encoder/eeg_imagenet40_cvpr_2017_raw/' # train or test
    print('loading dataset...')
    dataset = EEG_to_Image_dataset(cvpr40_test_path, make_class)
    dl = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=args.pin_mem, shuffle=True)
    save_folder = output_folder + make_class_name
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    if model_type=='transformer':
        # eeg_encoder = EEG_Transformer_model(hidden_size=128, projection_dim=256, num_layers=4).cuda()
        eeg_encoder = EEG_BiLSTM_original_model(hidden_size=256, projection_dim=256, num_layers=4).cuda()
    elif model_type=='lstm':
        eeg_encoder = EEG_LSTM_3_model(n_features=256, projection_dim=256, num_layers=4).cuda()
        # eeg_encoder = EEG_BiLSTM_model(n_features=256, projection_dim=256, num_layers=4).cuda()
    elif model_type=='cnn':
        eeg_encoder = CNN_model().cuda()

    if eegckpt is not None:
        eegcheckpoint = torch.load(eegckpt, map_location=torch.device("cuda"))
        eeg_encoder.load_state_dict(eegcheckpoint['model_state_dict'])
    else:
        print('There is no eeg chekpoint...')
    eeg_encoder.eval()
    image_generator.eval()
    to_pil = ToPILImage()
    moco_scores = []
    ssim_scores = []
    lpips_scores = []
    lpips_model = lpips.LPIPS(net='vgg')

    print('generating image...')
    idx = 1
    for image, eeg in tqdm(dl):
        goal_images = image
        with torch.no_grad():
            eeg = eeg.cuda()
            rep = eeg_encoder(eeg)
            # 標準化
            rep_std = torch.std(rep, dim=1, keepdim=True)
            rep_mean = torch.mean(rep, dim=1, keepdim=True)
            rep = (rep - rep_mean) / rep_std # eeg特徴量

            eeg_rep = rep

            gen_images_batch, _ = image_generator(goal_images, None, # 最初のNoneに画像を入れる必要がある。 goal_images_batch
                                        gen_image=True, bsz=batch_size,
                                        choice_temperature=args.temp,
                                        num_iter=args.num_iter, sampled_rep=None,
                                        rdm_steps=args.rdm_steps, eta=args.eta, cfg=cfg,
                                        eeg2image=True, eeg_rep=eeg_rep)
        gen_images_batch = misc.concat_all_gather(gen_images_batch) # 全てのGPUから結果を集約
        gen_images_batch = gen_images_batch.detach().cpu()

        # lpips
        resize_transform = transforms.Resize((224, 224))
        lpips_scores.append(lpips_model(goal_images, resize_transform(gen_images_batch[0].unsqueeze(0))).item())

        # save img
        goal_image = np.clip(goal_images[0].numpy().transpose([1, 2, 0]) * 255, 0, 255)
        goal_image = goal_image.astype(np.uint8)[:, :, ::-1]

        gen_image = np.clip(gen_images_batch[0].numpy().transpose([1, 2, 0]) * 255, 0, 255)
        gen_image = gen_image.astype(np.uint8)[:, :, ::-1]
        gen_image = cv2.resize(gen_image, (224, 224), interpolation=cv2.INTER_LINEAR)

        grid_img = cv2_grid([goal_image, gen_image], 1, 2)

        cv2.imwrite(
            os.path.join(save_folder, '{}.png'.format(idx)),
            grid_img)

        # ssim
        ssim_score = evaluate_ssim(goal_image, gen_image)
        ssim_scores.append(ssim_score)

        # moco score
        with torch.no_grad():
            if len(goal_images)==1:
                goal_rep = image_generator(goal_images, None,
                                gen_image=True, bsz=1,
                                choice_temperature=args.temp,
                                sampled_rep=None,
                                rdm_steps=args.rdm_steps, eta=args.eta, cfg=0, return_rep=True)
            gen_images_batch=gen_images_batch[0].unsqueeze(0)
            if len(gen_images_batch)==1:
                generated_rep = image_generator(gen_images_batch, None,
                                gen_image=True, bsz=1,
                                choice_temperature=args.temp,
                                sampled_rep=None,
                                rdm_steps=args.rdm_steps, eta=args.eta, cfg=0, return_rep=True)

            cosine_similarity = torch.nn.functional.cosine_similarity(
                                goal_rep.cpu().float(), generated_rep.cpu().float(), dim=-1)
        moco_score = cosine_similarity.item()
        moco_scores.append(moco_score)

        idx+=1

    print('lpips score', sum(lpips_scores)/len(lpips_scores))

    return moco_scores, ssim_scores, lpips_scores

def marge_clip(goal_image, gen_image):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    # 画像を前処理
    goal_image = preprocess(goal_image).unsqueeze(0).to(device)
    gen_image = preprocess(gen_image).unsqueeze(0).to(device)

    # CLIPのImage Encoderに入力
    with torch.no_grad():
        goal_image_emb = model.encode_image(goal_image)
        gen_image_emb = model.encode_image(gen_image)

    cosine_similarity = torch.nn.functional.cosine_similarity(
    goal_image_emb.cpu().float(), gen_image_emb.cpu().float(), dim=-1)

    return cosine_similarity

from skimage.metrics import structural_similarity as ssim
def evaluate_ssim(goal_image, gen_image):
    """
    metric_values = evaluation(
        org_img=goal_image,
        pred_img=gen_image,
        metrics=['ssim'],
    )
    """
    score, _ = ssim(goal_image, gen_image, full=True, channel_axis=-1)

    return score
    

def vq_evaluate(image_generator, output_folder, args, batch_size=16, log_writer=None, cfg=0.0, make_class=None, make_class_name=None, eegckpt=None, model_type=None):
    # moco v3の出力と近いベクトルをcodebook内から何番目まで特定しその重み付き平均をMAGEの条件とする
    cvpr40_test_path = 'eeg_encoder/eeg_imagenet40_cvpr_2017_raw/' # train or test
    print('loading dataset...')
    dataset = EEG_to_Image_dataset(cvpr40_test_path, make_class)
    dl = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=args.pin_mem, shuffle=True)
    save_folder = output_folder + make_class_name
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

    if model_type=='transformer':
        # eeg_encoder = EEG_Transformer_model(hidden_size=128, projection_dim=256, num_layers=4).cuda()
        eeg_encoder = EEG_Transformer_model(hidden_size=128, projection_dim=256, num_layers=3).cuda()
    elif model_type=='lstm':
        # eeg_encoder = EEG_LSTM_model(n_features=128, projection_dim=256, num_layers=4).cuda()
        eeg_encoder = EEG_LSTM_3_model(n_features=256, projection_dim=256, num_layers=3).cuda()

    if eegckpt is not None:
        eegcheckpoint = torch.load(eegckpt, map_location=torch.device("cuda"))
        eeg_encoder.load_state_dict(eegcheckpoint['model_state_dict'])
    else:
        print('There is no eeg chekpoint...')
    eeg_encoder.eval()
    image_generator.eval()
    to_pil = ToPILImage()
    moco_scores = []
    ssim_scores = []
    lpips_scores = []
    lpips_model = lpips.LPIPS(net='vgg')

    print('generating image...')
    for idx, (image, eeg) in enumerate(dl):
        print('generating: ', idx+1, '/', len(dl))
        goal_images = image
        with torch.no_grad():
            eeg = eeg.cuda()
            rep = eeg_encoder(eeg)
            
            codebook = mk_vq_code_book()
            
            print('searching near features...')
            vq_reps, distances = search_near_rep(rep, codebook, 4)
            print('vq distances: ', distances)

            # vq_repsの重み付き平均を得る
            max_value = torch.max(distances)
            difference = max_value - distances
            weights = difference / torch.sum(difference)
            print(weights)
            vq_reps = vq_reps.cuda()
            weighted_vectors = vq_reps * weights.unsqueeze(1)
            weighted_average = torch.sum(weighted_vectors, dim=0)
            print(weighted_average.shape)
            vq_rep = weighted_average.unsqueeze(0)

            # 標準化
            """
            rep_std = torch.std(vq_rep, dim=1, keepdim=True)
            rep_mean = torch.mean(vq_rep, dim=1, keepdim=True)
            vq_rep = (vq_rep - rep_mean) / rep_std # eeg特徴量
            """

            eeg_rep = vq_rep

            gen_images_batch, _ = image_generator(goal_images, None, # 最初のNoneに画像を入れる必要がある。 goal_images_batch
                                        gen_image=True, bsz=batch_size,
                                        choice_temperature=args.temp,
                                        num_iter=args.num_iter, sampled_rep=None,
                                        rdm_steps=args.rdm_steps, eta=args.eta, cfg=cfg,
                                        eeg2image=True, eeg_rep=eeg_rep)
        gen_images_batch = misc.concat_all_gather(gen_images_batch) # 全てのGPUから結果を集約
        gen_images_batch = gen_images_batch.detach().cpu()

        # lpips
        resize_transform = transforms.Resize((224, 224))
        lpips_scores.append(lpips_model(goal_images, resize_transform(gen_images_batch[0].unsqueeze(0))).item())

        # save img
        goal_image = np.clip(goal_images[0].numpy().transpose([1, 2, 0]) * 255, 0, 255)
        goal_image = goal_image.astype(np.uint8)[:, :, ::-1]

        gen_image = np.clip(gen_images_batch[0].numpy().transpose([1, 2, 0]) * 255, 0, 255)
        gen_image = gen_image.astype(np.uint8)[:, :, ::-1]
        gen_image = cv2.resize(gen_image, (224, 224), interpolation=cv2.INTER_LINEAR)

        grid_img = cv2_grid([goal_image, gen_image], 1, 2)

        cv2.imwrite(
            os.path.join(save_folder, '{}.png'.format(idx)),
            grid_img)

        # ssim
        ssim_score = evaluate_ssim(goal_image, gen_image)
        ssim_scores.append(ssim_score)
        
        # moco score
        with torch.no_grad():
            if len(goal_images)==1:
                goal_rep = image_generator(goal_images, None,
                                gen_image=True, bsz=1,
                                choice_temperature=args.temp,
                                sampled_rep=None,
                                rdm_steps=args.rdm_steps, eta=args.eta, cfg=0, return_rep=True)
            gen_images_batch=gen_images_batch[0].unsqueeze(0)
            if len(gen_images_batch)==1:
                generated_rep = image_generator(gen_images_batch, None,
                                gen_image=True, bsz=1,
                                choice_temperature=args.temp,
                                sampled_rep=None,
                                rdm_steps=args.rdm_steps, eta=args.eta, cfg=0, return_rep=True)

            cosine_similarity = torch.nn.functional.cosine_similarity(
                                goal_rep.cpu().float(), generated_rep.cpu().float(), dim=-1)
        moco_score = cosine_similarity.item()
        moco_scores.append(moco_score)

    print('lpips score', sum(lpips_scores)/len(lpips_scores))

    return moco_scores, ssim_scores

import torch.nn.functional as F
def search_near_rep(origin_rep, codebook, num):
    """
    codebook内からorigin_repとL2距離の近いものをnum個取ってくる
    """

    # L2 距離を計算
    distances = torch.cdist(origin_rep, codebook.cuda(), p=2)  # [1, N]

    distances = distances.squeeze(0)  # [N]

    # 最小距離のインデックスを取得
    closest_indices = torch.argsort(distances)[:num]

    # 最も近い特徴量を取得
    closest_features = codebook[closest_indices]
    closest_distances = distances[closest_indices]

    return closest_features, closest_distances


def mk_vq_code_book(dic=None):
    if dic is None:
        # trainファイルから辞書をロード
        label_path = '/workspace/simple_rcg/eeg_encoder/label_dic/imagenet_feature.pkl'

        # pklファイルからロードしてくる
        with open(label_path, 'rb') as f:
            label_dic = pickle.load(f)
    else:
        label_dic=dic

    codebook = []
    for feature in label_dic:
        codebook.append(feature)
    
    codebook=torch.stack(codebook)

    return codebook
