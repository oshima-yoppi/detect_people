# %%
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import glob
import matplotlib.pyplot as plt
import os

# モデルの読み込み
model_path = "model/model_mobile_good.h5"
model = tf.keras.models.load_model(model_path) 
pix = 224 
# %%
# 予測対象の画像を読み込む
all_img_dir = "ML_imgs" # 予測させたい画像が入っているディレクトリ
all_image_path = glob.glob(os.path.join(all_img_dir, "*.png")) + glob.glob(
    os.path.join(all_img_dir, "*.JPG")
) # ディレクトリ内の画像のパスを全て取得

save_dir = "result" # 予測結果を保存するディレクトリ
if not os.path.exists(save_dir): # そのディレクトリがなければ作成
    os.makedirs(save_dir)


for image_path in all_image_path:
    image = cv2.imread(image_path)
    # print(image.size)
    # 画像の前処理
    image = cv2.resize(image, ((pix, pix)))  # モデルの入力サイズにリサイズ
    image = np.array(image) / 255.0  # 画像を0~1の範囲に正規化
    image = np.expand_dims(image, axis=0)  # バッチ次元を追加
    # print(image.shape)

    # モデルの予測
    predictions = model.predict(image)
    # print(predictions.shape)
    predicted_class = np.argmax(predictions, axis=1)  # 最も確信度の高いクラスを取得
    print("hito" if predicted_class == 0 else "other")
    res = "hito" if predicted_class == 0 else "other"

    # 予測結果の表示
    print("予測結果:", predicted_class, predictions)
    pro = predictions[0]
    pro = np.round(pro, 3)[0]
    plt.figure()
    plt.title(f"Predicted: {res}, pro:{pro}")
    plt.imshow(image[0])
    print(image_path)
    plt.savefig(f"{save_dir}/{os.path.basename(image_path)}")
    # plt.show()

# %%
