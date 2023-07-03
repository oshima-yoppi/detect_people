import tensorflow as tf
import cv2

# データセットの読み込み
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    directory="hito_dataset",
    image_size=(224, 224),
    batch_size=1,
    shuffle=True,
    seed=42,
)

# クラスのラベルを取得
class_labels = dataset.class_names

# サンプルの表示
for images, labels in dataset.take(5):  # 5つのサンプルを取得
    image = images[0].numpy().astype("uint8")  # 画像データをNumpy配列に変換
    label = labels[0].numpy()  # ラベルを取得

    # 画像の表示
    cv2.imshow("Image", image)
    cv2.waitKey(0)

    # クラスラベルの表示
    class_name = class_labels[label]
    print("クラスラベル:", class_name)
