import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import pixel_generator.mage.models_mage as models_mage

from engine_mage import gen_img
from mk_label import gen_big_img_feat, gen_imagenet_feat, gen_small_img_feat
from image_to_image import image_to_image


def get_args_parser():
    parser = argparse.ArgumentParser('MAGE training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mage_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=256, type=int,
                        help='images input size')

    parser.add_argument('--vqgan_ckpt_path',  default='pretrained_vqgan/vqgan_jax_strongaug.ckpt', type=str)

    # Pre-trained enc parameters
    parser.add_argument('--use_rep', action='store_true', help='use representation as condition.')
    parser.add_argument('--use_class_label', action='store_true', help='use class label as condition.')
    parser.add_argument('--rep_dim', default=256, type=int)
    parser.add_argument('--pretrained_enc_arch',  default=None, type=str)
    parser.add_argument('--pretrained_enc_path',  default=None, type=str)
    parser.add_argument('--pretrained_enc_proj_dim',  default=256, type=int)
    parser.add_argument('--pretrained_enc_withproj', action='store_true')

    # RDM parameters
    parser.add_argument('--pretrained_rdm_ckpt',  default=None, type=str)
    parser.add_argument('--pretrained_rdm_cfg',  default=None, type=str)
    parser.add_argument('--rdm_steps', default=250, type=int)
    parser.add_argument('--eta', default=1.0, type=float)

    # Pixel generation parameters
    parser.add_argument('--evaluate', action='store_true', help="perform only evaluation")
    parser.add_argument('--eval_freq', type=int, default=40, help='evaluation frequency')
    parser.add_argument('--temp', default=6.0, type=float,
                        help='sampling temperature')
    parser.add_argument('--num_iter', default=16, type=int,
                        help='number of iterations for generation')
    parser.add_argument('--num_images', default=50000, type=int,
                        help='number of images to generate')
    parser.add_argument('--cfg', default=0.0, type=float)
    parser.add_argument('--rep_drop_prob', default=0.0, type=float)

    # MAGE params
    parser.add_argument('--mask_ratio_min', type=float, default=0.5,
                        help='Minimum mask ratio')
    parser.add_argument('--mask_ratio_max', type=float, default=1.0,
                        help='Maximum mask ratio')
    parser.add_argument('--mask_ratio_mu', type=float, default=0.55,
                        help='Mask ratio distribution peak')
    parser.add_argument('--mask_ratio_std', type=float, default=0.25,
                        help='Mask ratio distribution std')
    parser.add_argument('--grad_clip', type=float, default=3.0,
                        help='Gradient clip')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='./data/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('--augmentation', default='noaug', type=str,
                        help='Augmentation type')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args, ckpt_path=None, cfg=None):

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True


    # init log writer
    log_writer = None
 
    
    # define the model
    model = models_mage.__dict__[args.model](mask_ratio_mu=args.mask_ratio_mu, mask_ratio_std=args.mask_ratio_std,
                                             mask_ratio_min=args.mask_ratio_min, mask_ratio_max=args.mask_ratio_max,
                                             vqgan_ckpt_path=args.vqgan_ckpt_path,
                                             use_rep=args.use_rep,
                                             rep_dim=args.rep_dim,
                                             rep_drop_prob=args.rep_drop_prob,
                                             use_class_label=args.use_class_label,
                                             pretrained_enc_arch=args.pretrained_enc_arch,
                                             pretrained_enc_path=args.pretrained_enc_path,
                                             pretrained_enc_proj_dim=args.pretrained_enc_proj_dim,
                                             pretrained_enc_withproj=args.pretrained_enc_withproj,
                                             pretrained_rdm_ckpt=args.pretrained_rdm_ckpt,
                                             pretrained_rdm_cfg=args.pretrained_rdm_cfg)

    model.to(device)
    
    model_without_ddp = model
    
    resume_path = args.resume
    checkpoint = torch.load(resume_path, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])


    img2img=False
    if img2img:
        goal_image_path = '/workspace/simple_rcg/antelope.png'
        image_to_image(model_without_ddp, args, goal_image_path)
        return

    mk_label = True
    if mk_label:
        dataset_type = 'small'
        if dataset_type=='big':
            split_dic_path = '/workspace/CVPR2021-02785/preprocessed/imagenet40-1000-1_split.pth'
            data_path = '/workspace/CVPR2021-02785/preprocessed/imagenet40-1000-1.pth'
            stimuli_path = '/workspace/CVPR2021-02785/stimuli'
            split_lst = ['train', 'val', 'test']
            for split in split_lst:
                gen_big_img_feat(model_without_ddp, args, split_dic_path, data_path, stimuli_path, split)
            return
        if dataset_type=='small':
            split_lst = ['train', 'val', 'test']
            for split in split_lst:
                gen_small_img_feat(model_without_ddp, args, split)
            return


    class_lst = [ 'n02106662_dog', 'n02124075_Egyptian cat', 'n02281787_lycaenid', 'n02389026_sorrel', 'n02492035_capuchin',
                'n02504458_African elephant', 'n02510455_giantpanda', 'n02607072_anemonefish', 'n02690373_airliner', 'n02906734_broom',
                'n02951358_canoe', 'n02992529_cellulartelephone', 'n03063599_coffeemug', 'n03100240_convertible', 'n03180011_computer', 
                'n03197337_digitalwatch', 'n03272010_electric guitar', 'n03272562_electriclocomotive', 'n03297495_espressomaker', 
                'n03376595_foldingchair', 'n03445777_golfball', 'n03452741_grandpiano', 'n03584829_iron', "n03590841_jack-o'-lantern",
                'n03709823_mailbag', 'n03773504_missile', 'n03775071_mitten', 'n03792782_mountainbike', 'n03792972_mountaintent', 
                'n03877472_pajama', 'n03888257_parachute', 'n03982430_pooltable', 'n04044716_radiotelescope', 'n04069434_reflexcamera',
                'n04086273_revolver', 'n04120489_runningshoe', 'n07753592_banana', 'n07873807_pizza', 'n11939491_daisy', 'n13054560_bolete']

    # class_lst = ['n02106662_dog'] #, 'n02124075_Egyptian cat', 'n02281787_lycaenid', 'n02389026_sorrel', 'n02492035_capuchin']

    # 脳波からの画像生成
    eeg2image = True
    from engine_mage import eeg_to_image, evaluate, vq_evaluate

    if eeg2image:
        moco_scores = {}
        ssim_scores = {}
        lpips_scores = {}
        for element in class_lst:        
            eeg_ckpt = '/workspace/simple_rcg/eeg_encoder/unknown_model/EXPERIMENT_23_mse_lstm_all_train/per500_ckpts/epoch_250_0.16400782067280312.pth' 
            make_class = element.split('_')[0]
            make_class_name = element.split('_')[1]
            model_type = 'lstm' # 'transformer' or 'cnn' or 'lstm'
            output_folder = 'output_test_images_23_cfg6/' # output_unknown_class or output_known_class
            print('config: ', args.cfg)

            # 評価
            print(cfg)
            moco_score, ssim_score, lpips_score = evaluate(model_without_ddp, output_folder, args, cfg=6, make_class=make_class, make_class_name=make_class_name, eegckpt=eeg_ckpt, model_type=model_type)
            
            moco_scores[element] = sum(moco_score) / len(moco_score)
            print('moco score ', element, ': ', moco_score)
            print('moco score average ', element, ': ', sum(moco_score) / len(moco_score))

            ssim_scores[element] = sum(ssim_score) / len(ssim_score)
            print('ssim score ', element, ': ', ssim_score)
            print('ssim score average ', element, ': ', sum(ssim_score) / len(ssim_score))

            lpips_scores[element] = sum(lpips_score) / len(lpips_score)
            print('lpips score ', element, ': ', lpips_score)
            print('lpips score average ', element, ': ', sum(lpips_score) / len(lpips_score))
        
        print('_________________________________________________________________')
        print('each class score : ', moco_scores)
        print('_________________________________________________________________')
        print('moco score avarage :', sum(moco_scores.values())/len(moco_scores))
        print('_________________________________________________________________')
        print('each class score : ', ssim_scores)
        print('_________________________________________________________________')
        print('ssim score avarage :', sum(ssim_scores.values())/len(ssim_scores))
        print('_________________________________________________________________')
        print('each class score : ', lpips_scores)
        print('_________________________________________________________________')
        print('lpips score avarage :', sum(lpips_scores.values())/len(lpips_scores))

        return sum(lpips_scores.values())/len(lpips_scores)

    moco_sd=False
    if moco_sd:
        from evaluate_brain_decoder import evaluate_sd
        if moco_sd:
            evaluate_sd(model_without_ddp, args=args, cfg=args.cfg)
            return


    for element in class_lst: 
        dataset_path='eeg_encoder/eeg_imagenet40_cvpr_2017_raw/train/'

        print("Start evaluating")
        gen_img(model_without_ddp, args, 0, batch_size=16, log_writer=log_writer, cfg=0, make_class=element, dataset_path=dataset_path)
        # if args.cfg > 0:
            # gen_img(model_without_ddp, data_loader, args, 0, batch_size=16, log_writer=log_writer, cfg=args.cfg)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    args.log_dir = args.output_dir

    main(args)
