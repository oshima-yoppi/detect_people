# %%
import os
import numpy as np

from pycocotools.coco import COCO
from PIL import Image

# COCOデータセットのパスとアノテーションファイルのパス
data_dir = "coco_dataset"
annotation_file = os.path.join(data_dir, "annotations", "instances_val2017.json")

# COCOデータセットのロード
coco = COCO(annotation_file)
# %%
# print(coco, type(coco))
# 分類するカテゴリID（人の場合は1）
category_id = 1

# 出力ディレクトリのパス
train_dir_hito = "hito_dataset/train/hito"
train_dir_none = "hito_dataset/train/none"
test_dir_hito = "hito_dataset/test/hito"
test_dir_none = "hito_dataset/test/none"
os.makedirs(train_dir_hito, exist_ok=True)
os.makedirs(train_dir_none, exist_ok=True)
os.makedirs(test_dir_hito, exist_ok=True)
os.makedirs(test_dir_none, exist_ok=True)


# COCOデータセットの画像IDを取得
image_ids = coco.getImgIds()
# %%
len(image_ids)
# %%
# 画像ごとに処理
for i, image_id in enumerate(image_ids):
    # 画像情報の取得
    image_info = coco.loadImgs(image_id)[0]
    # print(image_info)
    image_path = os.path.join(data_dir, "val2017", image_info["file_name"])

    # 画像の読み込み
    image = Image.open(image_path)

    # 画像内に人が存在するかを判定
    annotation_ids = coco.getAnnIds(imgIds=image_id, catIds=category_id, iscrowd=None)
    annotations = coco.loadAnns(annotation_ids)
    has_person = len(annotations) > 0
    if i <= len(image_ids) * 0.8:
        # 人の存在を分類して画像を保存
        if has_person:
            save_path = os.path.join(train_dir_hito, f'{image_info["file_name"]}')
            image.save(save_path)
        else:
            save_path = os.path.join(train_dir_none, f'{image_info["file_name"]}')
            image.save(save_path)
    else:
        # 人の存在を分類して画像を保存
        if has_person:
            save_path = os.path.join(test_dir_hito, f'{image_info["file_name"]}')
            image.save(save_path)
        else:
            save_path = os.path.join(test_dir_none, f'{image_info["file_name"]}')
            image.save(save_path)

# %%
