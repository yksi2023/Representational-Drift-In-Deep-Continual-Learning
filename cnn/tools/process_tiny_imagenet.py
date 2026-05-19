import os
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def prepare_tiny_imagenet(raw_root="data/tiny-imagenet-200", out_root="data/tiny-imagenet-200-processed"):
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(os.path.join(out_root, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "val"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "test"), exist_ok=True)

    # ---------- 1. 拷贝训练集 ----------
    print("Copying training set...")
    train_src = os.path.join(raw_root, "train")
    for cls in tqdm(os.listdir(train_src)):
        cls_src = os.path.join(train_src, cls, "images")
        cls_dst = os.path.join(out_root, "train", cls)
        os.makedirs(cls_dst, exist_ok=True)
        for fname in os.listdir(cls_src):
            shutil.copy(os.path.join(cls_src, fname), os.path.join(cls_dst, fname))

    # ---------- 2. 重建验证集结构 ----------
    print("Reorganizing val set...")
    val_dir = os.path.join(raw_root, "val")
    ann_file = os.path.join(val_dir, "val_annotations.txt")
    with open(ann_file, "r") as f:
        lines = f.readlines()
    for line in tqdm(lines):
        img_name, cls = line.strip().split("\t")[:2]
        cls_dst = os.path.join(out_root, "val", cls)
        os.makedirs(cls_dst, exist_ok=True)
        src_path = os.path.join(val_dir, "images", img_name)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(cls_dst, img_name))

    # ---------- 3. 从训练集划分 test ----------
    # 为了有 test，我们从 train 里每个类抽 10% 作为 test
    print("Splitting test set from train...")
    for cls in tqdm(os.listdir(os.path.join(out_root, "train"))):
        cls_dir = os.path.join(out_root, "train", cls)
        imgs = os.listdir(cls_dir)
        train_imgs, test_imgs = train_test_split(imgs, test_size=0.1, random_state=42)
        test_dst = os.path.join(out_root, "test", cls)
        os.makedirs(test_dst, exist_ok=True)
        for img in test_imgs:
            shutil.move(os.path.join(cls_dir, img), os.path.join(test_dst, img))

    print(" Done! Output in", out_root)

if __name__ == "__main__":
    prepare_tiny_imagenet()
