#!/usr/bin/env python3
"""Prepare an ImageNet-1K N-class subset for incremental learning.

Input (raw_root):
  ILSVRC2012_img_train.tar       -- outer tar of 1000 inner class tars
  ILSVRC2012_img_val.tar         -- 50 000 flat JPEG images
  ILSVRC2012_devkit_t12.tar.gz   -- metadata (meta.mat + val ground truth)

Output (out_root):
  train/<wnid>/<image>.JPEG      -- ~1170 imgs/class after test split
  val/<wnid>/<image>.JPEG        -- 50 imgs/class (official val)
  test/<wnid>/<image>.JPEG       -- ~130 imgs/class (10 % split from train)
  selected_wnids.txt             -- ordered list of the N chosen wnids

Usage:
  python cnn/process_imagenet.py
  python cnn/process_imagenet.py --n_classes 200 --raw_root cnn/data --out_root cnn/data/imagenet1k-200-processed
  python cnn/process_imagenet.py --wnids_file my_wnids.txt   # custom class list

Dependencies:  scipy (for meta.mat parsing), tqdm
"""
import argparse
import io
import os
import random
import shutil
import tarfile
from typing import Dict, List, Set

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Devkit helpers
# ---------------------------------------------------------------------------

def _extract_devkit(raw_root: str) -> str:
    """Extract ILSVRC2012_devkit_t12.tar.gz and return its data/ directory."""
    devkit_tar = os.path.join(raw_root, "ILSVRC2012_devkit_t12.tar.gz")
    devkit_out = os.path.join(raw_root, "_devkit_extracted")
    if not os.path.isdir(devkit_out):
        print("Extracting devkit...")
        with tarfile.open(devkit_tar, "r:gz") as tf:
            tf.extractall(devkit_out)
    data_dir = os.path.join(devkit_out, "ILSVRC2012_devkit_t12", "data")
    return data_dir


def _load_id_to_wnid(data_dir: str) -> Dict[int, str]:
    """Return {1-indexed synset ID → wnid} from devkit meta.mat."""
    import scipy.io
    meta_path = os.path.join(data_dir, "meta.mat")
    meta = scipy.io.loadmat(meta_path, squeeze_me=True)["synsets"]
    id_to_wnid: Dict[int, str] = {}
    for entry in meta:
        sid = int(entry["ILSVRC2012_ID"])
        wnid = str(entry["WNID"]).strip()
        id_to_wnid[sid] = wnid
    return id_to_wnid


def _load_val_labels(data_dir: str) -> List[int]:
    """Return list of 1-indexed synset IDs for the 50 000 val images."""
    gt_path = os.path.join(data_dir, "ILSVRC2012_validation_ground_truth.txt")
    with open(gt_path) as f:
        return [int(line.strip()) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Class selection
# ---------------------------------------------------------------------------

def _select_wnids(train_tar_path: str, n_classes: int, wnids_file: str = None) -> List[str]:
    """Return ordered list of selected wnids."""
    if wnids_file and os.path.isfile(wnids_file):
        with open(wnids_file) as f:
            wnids = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(wnids)} wnids from {wnids_file}")
        return wnids

    print("Listing train tar to discover all wnids...")
    with tarfile.open(train_tar_path, "r") as tf:
        all_wnids = sorted(
            m.name.split(".")[0]
            for m in tf.getmembers()
            if m.name.endswith(".tar")
        )
    if n_classes > len(all_wnids):
        raise ValueError(f"Requested {n_classes} classes but train tar has only {len(all_wnids)}")
    selected = all_wnids[:n_classes]
    print(f"Selected first {n_classes} wnids (alphabetical): {selected[:5]} ...")
    return selected


# ---------------------------------------------------------------------------
# Extraction steps
# ---------------------------------------------------------------------------

def _extract_train(train_tar_path: str, selected: Set[str], out_train: str) -> None:
    print("Extracting train images (streaming, selected classes only)...")
    with tarfile.open(train_tar_path, "r") as outer:
        members = [m for m in outer.getmembers()
                   if m.name.split(".")[0] in selected and m.name.endswith(".tar")]
        for member in tqdm(members, desc="train classes"):
            wnid = member.name.split(".")[0]
            cls_dir = os.path.join(out_train, wnid)
            if os.path.isdir(cls_dir) and os.listdir(cls_dir):
                continue  # already extracted
            os.makedirs(cls_dir, exist_ok=True)
            inner_bytes = outer.extractfile(member).read()
            with tarfile.open(fileobj=io.BytesIO(inner_bytes)) as inner:
                inner.extractall(path=cls_dir)


def _extract_val(val_tar_path: str, selected: Set[str],
                 id_to_wnid: Dict[int, str], val_labels: List[int],
                 out_val: str) -> None:
    print("Extracting val images (streaming, selected classes only)...")
    # Build index: image number (1-indexed) → wnid, only for selected
    selected_val_map: Dict[str, str] = {}  # filename → wnid
    for img_num, sid in enumerate(val_labels, start=1):
        wnid = id_to_wnid.get(sid, "")
        if wnid in selected:
            fname = f"ILSVRC2012_val_{img_num:08d}.JPEG"
            selected_val_map[fname] = wnid

    with tarfile.open(val_tar_path, "r") as tf:
        members = [m for m in tf.getmembers() if m.name in selected_val_map]
        for member in tqdm(members, desc="val images"):
            wnid = selected_val_map[member.name]
            cls_dir = os.path.join(out_val, wnid)
            os.makedirs(cls_dir, exist_ok=True)
            dst = os.path.join(cls_dir, member.name)
            if os.path.isfile(dst):
                continue
            img_data = tf.extractfile(member).read()
            with open(dst, "wb") as f:
                f.write(img_data)


def _split_test(out_train: str, out_test: str, test_ratio: float, seed: int) -> None:
    print("Splitting test set from train (10 % per class)...")
    rng = random.Random(seed)
    for wnid in tqdm(os.listdir(out_train)):
        cls_dir = os.path.join(out_train, wnid)
        if not os.path.isdir(cls_dir):
            continue
        imgs = sorted(os.listdir(cls_dir))
        n_test = max(1, int(len(imgs) * test_ratio))
        test_imgs = rng.sample(imgs, n_test)
        test_dir = os.path.join(out_test, wnid)
        os.makedirs(test_dir, exist_ok=True)
        for img in test_imgs:
            src = os.path.join(cls_dir, img)
            dst = os.path.join(test_dir, img)
            if not os.path.isfile(dst):
                shutil.move(src, dst)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare_imagenet(
    raw_root: str = "cnn/data",
    out_root: str = "cnn/data/imagenet1k-200-processed",
    n_classes: int = 200,
    test_ratio: float = 0.1,
    seed: int = 42,
    wnids_file: str = None,
) -> None:
    train_tar = os.path.join(raw_root, "ILSVRC2012_img_train.tar")
    val_tar   = os.path.join(raw_root, "ILSVRC2012_img_val.tar")

    for p in (train_tar, val_tar):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing: {p}")

    out_train = os.path.join(out_root, "train")
    out_val   = os.path.join(out_root, "val")
    out_test  = os.path.join(out_root, "test")
    for d in (out_train, out_val, out_test):
        os.makedirs(d, exist_ok=True)

    # 1. Devkit
    data_dir   = _extract_devkit(raw_root)
    id_to_wnid = _load_id_to_wnid(data_dir)
    val_labels = _load_val_labels(data_dir)

    # 2. Select classes
    selected_list = _select_wnids(train_tar, n_classes, wnids_file)
    selected_set  = set(selected_list)

    wnids_out = os.path.join(out_root, "selected_wnids.txt")
    with open(wnids_out, "w") as f:
        f.write("\n".join(selected_list) + "\n")
    print(f"Saved class list → {wnids_out}")

    # 3. Train
    _extract_train(train_tar, selected_set, out_train)

    # 4. Val
    _extract_val(val_tar, selected_set, id_to_wnid, val_labels, out_val)

    # 5. Test split
    _split_test(out_train, out_test, test_ratio, seed)

    # Summary
    n_train = sum(len(os.listdir(os.path.join(out_train, w))) for w in selected_list)
    n_val   = sum(len(os.listdir(os.path.join(out_val, w)))
                  for w in selected_list if os.path.isdir(os.path.join(out_val, w)))
    n_test  = sum(len(os.listdir(os.path.join(out_test, w)))
                  for w in selected_list if os.path.isdir(os.path.join(out_test, w)))
    print(f"\nDone! Output → {out_root}")
    print(f"  train: {n_train}  val: {n_val}  test: {n_test}  classes: {n_classes}")


def _parse_args():
    p = argparse.ArgumentParser(description="Prepare ImageNet-1K N-class subset")
    p.add_argument("--raw_root",   default="cnn/data",
                   help="Directory containing the three ILSVRC2012 tar files")
    p.add_argument("--out_root",   default="cnn/data/imagenet1k-200-processed",
                   help="Output directory for processed dataset")
    p.add_argument("--n_classes",  type=int, default=200,
                   help="Number of classes to keep (first N wnids alphabetically)")
    p.add_argument("--test_ratio", type=float, default=0.1,
                   help="Fraction of train images to move to test split")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--wnids_file", default=None,
                   help="Optional text file (one wnid per line) to override class selection")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare_imagenet(
        raw_root=args.raw_root,
        out_root=args.out_root,
        n_classes=args.n_classes,
        test_ratio=args.test_ratio,
        seed=args.seed,
        wnids_file=args.wnids_file,
    )
