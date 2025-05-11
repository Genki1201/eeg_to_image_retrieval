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
import pickle

from eegdataset import EEG_to_Image_dataset
from eeg_encoder.network import EEG_Transformer_model, EEG_LSTM_model, CNN_model
from torchvision.transforms import ToPILImage
import clip
from torchvision import transforms



def split_image(img):
    if isinstance(img, np.ndarray):  # OpenCV (cv2) 画像の場合
        # Get the height and width
        height, width, _ = img.shape

        # Calculate the middle point
        middle = width // 2

        # Split the image into left and right halves
        goal = img[:, :middle, :]
        generated = img[:, middle:, :]

    elif isinstance(img, Image.Image):  # Pillow 画像の場合
        # Get the width and height
        width, height = img.size

        # Calculate the middle point
        middle = width // 2

        # Split the image into left and right halves
        goal = img.crop((0, 0, middle, height))  # 左半分
        generated = img.crop((middle, 0, width, height))  # 右半分

    else:
        raise TypeError("Unsupported image type. Use PIL.Image.Image or numpy.ndarray.")

    return goal, generated


from ssim.main import evaluation
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

def cv2_to_tensor(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype('float32') / 255.0
    image_chw = image_rgb.transpose(2, 0, 1)
    image_tensor = torch.tensor(image_chw)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor

def evaluate_sd(image_generator, args, batch_size=16, log_writer=None, cfg=0.0, make_class=None, make_class_name=None, eegckpt=None, model_type=None):
    # stable diffusion のmoco_score
    class_lst = [ 'n02106662_dog', 'n02124075_Egyptian cat', 'n02281787_lycaenid', 'n02389026_sorrel', 'n02492035_capuchin',
                'n02504458_African elephant', 'n02510455_giantpanda', 'n02607072_anemonefish', 'n02690373_airliner', 'n02906734_broom',
                'n02951358_canoe', 'n02992529_cellulartelephone', 'n03063599_coffeemug', 'n03100240_convertible', 'n03180011_computer', 
                'n03197337_digitalwatch', 'n03272010_electric guitar', 'n03272562_electriclocomotive', 'n03297495_espressomaker', 
                'n03376595_foldingchair', 'n03445777_golfball', 'n03452741_grandpiano', 'n03584829_iron', "n03590841_jack-o'-lantern",
                'n03709823_mailbag', 'n03773504_missile', 'n03775071_mitten', 'n03792782_mountainbike', 'n03792972_mountaintent', 
                'n03877472_pajama', 'n03888257_parachute', 'n03982430_pooltable', 'n04044716_radiotelescope', 'n04069434_reflexcamera',
                'n04086273_revolver', 'n04120489_runningshoe', 'n07753592_banana', 'n07873807_pizza', 'n11939491_daisy', 'n13054560_bolete']
    
    sd_image_path = '/workspace/IP-Adapter/eeg_to_image_output/test_image/' # image_jack-o'-lantern_with_text

    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # リサイズ (例: 128x128)
        transforms.ToTensor(),          # Tensorに変換

    ])

    all_moco_scores = {}
    all_ssim_scores = {}

    for c in class_lst: # クラスのループ
        moco_scores = []
        ssim_scores = []
        folder = sd_image_path + 'image_' + c.split('_')[1] + '_with_text/'
        print(folder)
        for file_name in tqdm(os.listdir(folder)): # 画像のループ
            # 拡張子が.pngのファイルのみ処理
            if file_name.endswith(".png"):
                file_path = os.path.join(folder, file_name)  # フルパスを生成
            image = Image.open(file_path)  # 画像をPillowとして読み込み
            goal_image, generated_image = split_image(image)
            goal_images = preprocess(goal_image).unsqueeze(0)
            gen_images_batch = preprocess(generated_image).unsqueeze(0)

            with torch.no_grad():
                if len(goal_images)==1:
                    goal_rep = image_generator(goal_images, None,
                                    gen_image=True, bsz=1,
                                    choice_temperature=args.temp,
                                    sampled_rep=None,
                                    rdm_steps=args.rdm_steps, eta=args.eta, cfg=0, return_rep=True)
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

            # ssim
            cv2_image = cv2.imread(file_path)
            cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
            cv2_goal_image, cv2_generated_image = split_image(cv2_image)
            # print(cv2_generated_image.shape) 224

            ssim_score = evaluate_ssim(cv2_generated_image, cv2_goal_image)
            ssim_scores.append(ssim_score)

        # print('moco ', c, ': ', moco_scores)
        all_moco_scores[c] = (sum(moco_scores) / len(moco_scores))

        # print('moco ', c, ': ', ssim_scores)
        all_ssim_scores[c] = (sum(ssim_scores) / len(ssim_scores))


    print('_______________________________________________________________')
    print('each moco score', all_moco_scores)
    print('_______________________________________________________________')
    print('moco test score avarage :', sum(all_moco_scores.values())/len(all_moco_scores)) # 0.4047265755335139
    print('_______________________________________________________________')
    print('each ssim score', all_ssim_scores)
    print('_______________________________________________________________')
    print('ssim test score avarage :', sum(all_ssim_scores.values())/len(all_ssim_scores))

if __name__=='__main__':
    """
    # 辞書データ
    mage = {
        'n02106662_dog': 0.7256216005637095, 
        'n02124075_Egyptian cat': 0.7057170625776052, 
        'n02281787_lycaenid': 0.7184467698846545, 
        'n02389026_sorrel': 0.6366605290344783, 
        'n02492035_capuchin': 0.6988242007791996, 
        'n02504458_African elephant': 0.7329800114035606, 
        'n02510455_giantpanda': 0.7468010468615426, 
        'n02607072_anemonefish': 0.6766959031422933, 
        'n02690373_airliner': 0.5858284555601351, 
        'n02906734_broom': 0.5474807633595034, 
        'n02951358_canoe': 0.6063133890430132, 
        'n02992529_cellulartelephone': 0.5449882022760533, 
        'n03063599_coffeemug': 0.5956877157801673, 
        'n03100240_convertible': 0.5756799644894071, 
        'n03180011_computer': 0.6289687930709786, 
        'n03197337_digitalwatch': 0.6101630267997583, 
        'n03272010_electric guitar': 0.5971488295881836, 
        'n03272562_electriclocomotive': 0.48574789199564194, 
        'n03297495_espressomaker': 0.6561822030279372, 
        'n03376595_foldingchair': 0.6341648058344921, 
        'n03445777_golfball': 0.622632476946582, 
        'n03452741_grandpiano': 0.5779966180523236, 
        'n03584829_iron': 0.5990021452307701, 
        "n03590841_jack-o'-lantern": 0.6548430433979741, 
        'n03709823_mailbag': 0.611519967064713, 
        'n03773504_missile': 0.5709370821714401, 
        'n03775071_mitten': 0.6323614310134541, 
        'n03792782_mountainbike': 0.5793208530393698, 
        'n03792972_mountaintent': 0.5954673768331608, 
        'n03877472_pajama': 0.594718127858405, 
        'n03888257_parachute': 0.6356229605498137, 
        'n03982430_pooltable': 0.6353462268908818, 
        'n04044716_radiotelescope': 0.6186614415297905, 
        'n04069434_reflexcamera': 0.6253487622986237, 
        'n04086273_revolver': 0.6184404343366623, 
        'n04120489_runningshoe': 0.5859080644117461, 
        'n07753592_banana': 0.6182162179833367, 
        'n07873807_pizza': 0.6411203675799899, 
        'n11939491_daisy': 0.6843073291308952, 
        'n13054560_bolete': 0.6317496369493768
    }

    sd = {
        'n02106662_dog': 0.7385044086437959, 
        'n02124075_Egyptian cat': 0.728389261290431, 
        'n02281787_lycaenid': 0.6822373029731569, 
        'n02389026_sorrel': 0.658897802943275, 
        'n02492035_capuchin': 0.707156045983235, 
        'n02504458_African elephant': 0.7746130704879761, 
        'n02510455_giantpanda': 0.8351446357038286, 
        'n02607072_anemonefish': 0.7201449910799662, 
        'n02690373_airliner': 0.6042280287453623, 
        'n02906734_broom': 0.5523361337907386, 
        'n02951358_canoe': 0.600916946431001, 
        'n02992529_cellulartelephone': 0.5631049512713043, 
        'n03063599_coffeemug': 0.5998960222516742, 
        'n03100240_convertible': 0.5607763164573245, 
        'n03180011_computer': 0.6507773254480627, 
        'n03197337_digitalwatch': 0.6339739362398783, 
        'n03272010_electric guitar': 0.5977289638033619, 
        'n03272562_electriclocomotive': 0.5481240728663074, 
        'n03297495_espressomaker': 0.6640077514780892, 
        'n03376595_foldingchair': 0.6549514140933752, 
        'n03445777_golfball': 0.6423141360282898, 
        'n03452741_grandpiano': 0.6243884553511937, 
        'n03584829_iron': 0.619936142116785, 
        "n03590841_jack-o'-lantern": 0.6487399461092772, 
        'n03709823_mailbag': 0.6166217227776846, 
        'n03773504_missile': 0.5869328967399068, 
        'n03775071_mitten': 0.6145781472776876, 
        'n03792782_mountainbike': 0.5937518250134032, 
        'n03792972_mountaintent': 0.6555906323095163, 
        'n03877472_pajama': 0.581352575152528, 
        'n03888257_parachute': 0.6612470133437051, 
        'n03982430_pooltable': 0.664814471701781, 
        'n04044716_radiotelescope': 0.6438921503722668, 
        'n04069434_reflexcamera': 0.6242381601283947, 
        'n04086273_revolver': 0.611906103293101, 
        'n04120489_runningshoe': 0.5967482800285021, 
        'n07753592_banana': 0.6111903587977091, 
        'n07873807_pizza': 0.6266406647585057, 
        'n11939491_daisy': 0.6917312398101344, 
        'n13054560_bolete': 0.691120797649343
    }

    class_lst = [ 'n02106662_dog', 'n02124075_Egyptian cat', 'n02281787_lycaenid', 'n02389026_sorrel', 'n02492035_capuchin',
                    'n02504458_African elephant', 'n02510455_giantpanda', 'n02607072_anemonefish', 'n02690373_airliner', 'n02906734_broom',
                    'n02951358_canoe', 'n02992529_cellulartelephone', 'n03063599_coffeemug', 'n03100240_convertible', 'n03180011_computer', 
                    'n03197337_digitalwatch', 'n03272010_electric guitar', 'n03272562_electriclocomotive', 'n03297495_espressomaker', 
                    'n03376595_foldingchair', 'n03445777_golfball', 'n03452741_grandpiano', 'n03584829_iron', "n03590841_jack-o'-lantern",
                    'n03709823_mailbag', 'n03773504_missile', 'n03775071_mitten', 'n03792782_mountainbike', 'n03792972_mountaintent', 
                    'n03877472_pajama', 'n03888257_parachute', 'n03982430_pooltable', 'n04044716_radiotelescope', 'n04069434_reflexcamera',
                    'n04086273_revolver', 'n04120489_runningshoe', 'n07753592_banana', 'n07873807_pizza', 'n11939491_daisy', 'n13054560_bolete']

    marge_dic = {}
    for c in class_lst:
        # print(c, ': ', mage[c] - sd[c])
        marge_dic[c] = mage[c] - sd[c]

    sorted_items = sorted(marge_dic.items(), key=lambda item: item[1])
    for item in sorted_items:
        print(item)
    """
