import json
import os
from PIL import Image


stimuli_path = '/workspace/CVPR2021-02785/stimuli'
image_data = '/workspace/eeg_to_image_rcg/eeg_encoder/correct_image_name_k5_moco.json'

target_size = (562, 562)  
space = 50

with open(image_data, "r", encoding="utf-8") as f:
    data = json.load(f)

for key in data.keys():
    images = []
    img_path = os.path.join(stimuli_path, key + '.JPEG')
    img = Image.open(img_path).convert("RGB")
    resized_img = img.resize(target_size, Image.BICUBIC) 
    images.append(resized_img)

    for element in data[key]: 
        img_path = os.path.join(stimuli_path, element + '.JPEG')
        img = Image.open(img_path).convert("RGB")
        resized_img = img.resize(target_size, Image.BICUBIC) 
        images.append(resized_img)


    # 横に並べるための合計サイズを計算
    num_images = len(images)
    total_width = target_size[0] * num_images + space * (num_images - 1)
    total_height = target_size[1]

    # 白背景で新しい画像を作成
    new_image = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))

    # 貼り付け処理
    x_offset = 0
    for img in images:
        new_image.paste(img, (x_offset, 0))
        x_offset += target_size[0] + space

    # 保存 or 表示
    new_image.save(f"/workspace/eeg_to_image_rcg/eeg_encoder/correct_samples/moco/images_{key}.jpg")