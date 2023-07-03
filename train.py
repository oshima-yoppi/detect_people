# %%
import tensorflow as tf

# データセットのディレクトリとパラメータの設定
train_dir = "hito_dataset/train"
test_dir = "hito_dataset/test"
image_size = (300, 300)
batch_size = 32

# ImageDataGeneratorを使用してデータセットを読み込む
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,  # 画像を0から1の範囲に正規化
    shear_range=0.2,  # 画像をランダムにシアー変換
    zoom_range=0.2,  # 画像をランダムに拡大・縮小
    horizontal_flip=True,
)  # 画像をランダムに水平反転

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=image_size, batch_size=batch_size, class_mode="binary"
)  # 二値分類の場合は'class_mode'を'binary'に設定

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=image_size, batch_size=batch_size, class_mode="binary"
)

# %%

# モデルの構築
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            32, (3, 3), activation="relu", input_shape=(300, 300, 3)
        ),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation="relu"),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation="relu"),
        # tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation="sigmoid"),  # 二値分類の場合は出力層のユニット数を1にする
    ]
)

# モデルのコンパイル

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)


# クラスの重みを設定
class_weights = {0: 0.853, 1: 1.170}
# モデルの学習
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,
    validation_data=test_generator,
    validation_steps=len(test_generator),
    class_weight=class_weights,
)

# %%
model.save("model.h5")

# %%
