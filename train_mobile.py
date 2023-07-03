# %%
import tensorflow as tf
from keras.layers import Dense, Input, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
from keras.models import Model
from tensorflow.keras.applications import MobileNet
import tensorflow_model_optimization as tfmot

# データセットのディレクトリとパラメータの設定
train_dir = "hito_dataset/train"
test_dir = "hito_dataset/test"
# 入力する画像サイズ
pix = 224
image_size = (pix, pix)
# バッチサイズ
batch_size = 32

# ImageDataGeneratorを使用してデータセットを読み込む
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,  # 画像を0から1の範囲に正規化
    shear_range=0.2,  # 画像をランダムにシアー変換
    zoom_range=0.2,  # 画像をランダムに拡大・縮小
    horizontal_flip=True,# 画像をランダムに水平反転
)  

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=image_size, batch_size=batch_size, class_mode="categorical"
)  # 二値分類の場合は'class_mode'を'binary'に設定

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=image_size, batch_size=batch_size, class_mode="categorical"
)

# %%



# モデルの構築
# base model としてモバイルネットを採用。
base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(pix, pix, 3))
base_model.trainable = False# モバイルネットの重みを固定
model = tf.keras.models.Sequential(
    [
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(), # モバイルネット部分
        tf.keras.layers.Dense(128, activation="relu"), # ここからは転移学習部分
        tf.keras.layers.Dense(2, activation="softmax"),# ２クラス分類をするから最期は必ず２にする
    ]
)


optimizer = Adam(lr=0.0001)
model.compile(
    loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)

# 構築したモデルを確認
model.summary()

# クラスの重みを設定
class_weights = {0: 0.853, 1: 1.170}
# 訓練実行
model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=5, # 学習する回数。多いほど精度は上がるが、時間がかかる。
    batch_size=batch_size,
    class_weight=class_weights,
)

# %%
tfmodel_path = "model_mobile_good.h5"
model.save(tfmodel_path) # modelを保存する

# %%

# TensorFlow Liteモデルに変換
def convert_tfmodel(tfmodel_path, tflite_model_path='model_mobile.tflite'):
    """
    tensorflowモデルをtfliteモデルに変換する関数
    """
    # TensorFlowモデルを読み込む
    loaded_model = tf.keras.models.load_model(tfmodel_path)

    # TensorFlow Lite Converterを作成
    converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)

    # 変換オプションを設定（例: オプティマイズ、量子化）
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    # TensorFlow Liteモデルに変換
    tflite_model = converter.convert()

    # TensorFlow Liteモデルを保存
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)
    return

convert_tfmodel(tfmodel_path)

