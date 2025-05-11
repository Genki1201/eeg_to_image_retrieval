# 論文用に画像をgrid化する

from PIL import Image


def split_image(img):
    # Get the width and height of the image
    width, height = img.size

    # Calculate the middle point
    middle = width // 2

    # Split the image into left and right halves
    goal = img.crop((0, 0, middle, height))  # Left half
    generated = img.crop((middle, 0, width, height))  # Right half

    return goal, generated

def image_grid(imgs, rows, cols, padding_x=10, padding_y=25, bg_color=(255, 255, 255)):

    assert len(imgs) == rows * cols, "画像の数がrows * colsに一致する必要があります。"

    # 各画像の幅と高さを取得
    w, h = imgs[0].size

    # グリッド全体の幅と高さを計算
    grid_width = cols * w + (cols - 1) * padding_x
    grid_height = rows * h + (rows - 1) * padding_y

    # グリッド用のキャンバスを作成
    grid = Image.new('RGB', size=(grid_width, grid_height), color=bg_color)

    # グリッドに画像を貼り付け
    for i, img in enumerate(imgs):
        x = (i % cols) * (w + padding_x)  # 横方向の座標（余白を考慮）
        y = (i // cols) * (h + padding_y)  # 縦方向の座標（余白を考慮）
        grid.paste(img, box=(x, y))

    return grid

def concat(sd_path, mage_path):
    # サンプル画像を読み込む
    sd_image = Image.open(sd_path)
    mage_image = Image.open(mage_path)

    goal_image, sd_image = split_image(sd_image)
    _, mage_image = split_image(mage_image)

    images = [goal_image, sd_image, mage_image]
    grid = image_grid(images, rows=1, cols=3) 

    return grid


if __name__=='__main__':
    """
    # mageの画像とsdの画像をくっつける
    mage = '/workspace/simple_rcg/test_images_sample/jack/8.png'
    sd = '/workspace/IP-Adapter/eeg_to_image_output/test_image/image_jack/9.png'

    image = concat(mage, sd)
    image.save("samples/jack_42.jpg")

    # tent fish car jack butterfly train plane

    """
    # 一度分けて　mageとstableをくっつけて　最後に全体をくっつける 空白を倍に
    cat_0 = Image.open('/workspace/simple_rcg/samples/fish_14.jpg')
    cat_35 = Image.open('/workspace/simple_rcg/samples/fish_21.jpg')
    elephant_27 = Image.open('/workspace/simple_rcg/samples/car_3.jpg')
    elephant_41 = Image.open('/workspace/simple_rcg/samples/car_48.jpg')
    flower_31 = Image.open('/workspace/simple_rcg/samples/butterfly_0.jpg')
    flower_53 = Image.open('/workspace/simple_rcg/samples/butterfly_31.jpg')
    horse_0 = Image.open('/workspace/simple_rcg/samples/train_23.jpg')
    horse_13 = Image.open('/workspace/simple_rcg/samples/train_33.jpg')
    mashroom_2 = Image.open('/workspace/simple_rcg/samples/jack_8.jpg')
    mashroom_4 = Image.open('/workspace/simple_rcg/samples/jack_42.jpg')
    panda_0 = Image.open('/workspace/simple_rcg/samples/plane_18.jpg')
    panda_10 = Image.open('/workspace/simple_rcg/samples/plane_14.jpg')


    cat_elephant = [cat_0, cat_35, elephant_27, elephant_41]
    cat_elephant = image_grid(cat_elephant, rows=4, cols=1)
    cat_elephant.save("samples/fish_car.jpg") 

    flower_horse = [flower_31, flower_53, horse_0, horse_13]
    flower_horse = image_grid(flower_horse, rows=4, cols=1)
    flower_horse.save("samples/butterfly_train.jpg")

    mashroom_panda = [mashroom_2, mashroom_4, panda_0, panda_10]
    mashroom_panda = image_grid(mashroom_panda, rows=4, cols=1)
    mashroom_panda.save("samples/jack_plane.jpg") 

    all_sample = [cat_elephant, flower_horse, mashroom_panda]
    all_sample = image_grid(all_sample, rows=1, cols=3, padding_x=45)
    all_sample.save("samples/all_sample_2.jpg")
