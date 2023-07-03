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
pix = 128
image_size = (pix, pix)
batch_size = 32

# ImageDataGeneratorを使用してデータセットを読み込む
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,  # 画像を0から1の範囲に正規化
    shear_range=0.2,  # 画像をランダムにシアー変換
    zoom_range=0.2,  # 画像をランダムに拡大・縮小
    horizontal_flip=True,
)  # 画像をランダムに水平反転

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=image_size, batch_size=batch_size, class_mode="categorical"
)  # 二値分類の場合は'class_mode'を'binary'に設定

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=image_size, batch_size=batch_size, class_mode="categorical"
)

# %%
# 事前学習済みVGGNetの重みをロード
# VGG16のモデルと重みをインポート


model_path = "model_mobile_good.h5"
model = tf.keras.models.load_model(model_path)
# 構築したモデルを確認
model.summary()

# 量子化の適用
quantize_apply = tfmot.quantization.keras.quantize_apply
quantized_model = quantize_apply(model)

# クラスの重みを設定
class_weights = {0: 0.853, 1: 1.170}
# 量子化済みモデルの評価
quantized_model.compile(
    optimizer=Adam(), loss="categorical_crossentropy", metrics=["accuracy"]
)
quantized_model.evaluate(test_generator)
# 訓練実行
quantized_model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=5,
    batch_size=batch_size,
    class_weight=class_weights,
)

# %%
model.save("model_mobile_bad.h5")

# %%
