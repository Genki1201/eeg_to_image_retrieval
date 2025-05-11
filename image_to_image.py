import sys
from typing import Iterable
import os

import torch
from torchvision import transforms

import util.misc as misc
import util.lr_sched as lr_sched
import cv2
import torch_fidelity
import numpy as np
import shutil


# このimage to imageによって新しいデータセットの画像生成ができるかを確認できる
def image_to_image(model, args, goal_image_path, batch_size=1, log_writer=None, cfg=0.0): # goalimageのdataloaderが入ってくるようにする
    model.eval()

    save_folder = "Image2Image_test/"
    if misc.get_rank() == 0:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
   
    image = cv2.imread(goal_image_path) 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = image/255.0
    image = np.transpose(image, (2, 0, 1))
    image = torch.from_numpy(image).float()
    image = image.unsqueeze(0).cuda()
    print(image.shape)
 
    with torch.no_grad():
        gen_images_batch, _ = model(image, None, # 最初のNoneに画像を入れる必要がある。 goal_images_batch
                                    gen_image=True, bsz=batch_size,
                                    choice_temperature=args.temp,
                                    num_iter=args.num_iter, sampled_rep=None,
                                    rdm_steps=args.rdm_steps, eta=args.eta, cfg=10.0)
    gen_images_batch = misc.concat_all_gather(gen_images_batch) # 全てのGPUから結果を集約
    gen_images_batch = gen_images_batch.detach().cpu()
    image = image.detach().cpu()

    # save img
    print('saveing image...')
    for b_id in range(gen_images_batch.size(0)):
        gen_img = np.clip(gen_images_batch[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255)
        gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(
            os.path.join(save_folder, '{}.png'.format(str(0).zfill(5))),
            gen_img)
        print(image.shape)
        goal_img = np.clip(image[0].numpy().transpose([1, 2, 0]) * 255, 0, 255)
        goal_img = goal_img.astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(
                os.path.join(save_folder, '{}.png'.format(str(0).zfill(5) + '_goal')),
                goal_img)
    print('completed!')
