"""Prepare a 200-class ImageNet-21k-P subset that is **disjoint** from
ImageNet-1k, laid out for continual learning (ImageFolder compatible).

The script works in three phases; any can be re-run independently:

  1. scan    — stream the tarball once, record per-wnid image counts.
               Cached to ``<out_dir>/meta/scan_cache.json`` (slow, ~15-45 min
               on a 165 GB .tar.gz depending on disk).
  2. select  — filter out the 1000 IN1k wnids, enforce a minimum number of
               samples per class, stratify by wnid-offset bucket, then take
               ``--num_classes`` (default 200) deterministically via ``--seed``.
               Writes ``<out_dir>/meta/selected_wnids.txt`` and
               ``<out_dir>/meta/selection_meta.json``.
  3. extract — stream the tarball a second time, extracting only the selected
               classes. For each class, the tar's ``imagenet21k_train`` split
               is re-split 90/10 into our {train, val}; the tar's
               ``imagenet21k_val`` becomes our {test}. Deterministic per-file
               hashing decides each image's bucket.

Output tree (after phase 3)::

    <out_dir>/
      train/<wnid>/*.JPEG
      val/<wnid>/*.JPEG
      test/<wnid>/*.JPEG
      meta/
        scan_cache.json
        selected_wnids.txt
        selection_meta.json

Usage (all phases)::

    python cnn/tools/prepare_imagenet21k_p200.py \\
        --tar     E:/imagenet21k_resized.tar.gz \\
        --out_dir E:/imagenet21k_p200

Skip a phase with ``--skip_scan`` / ``--skip_select`` / ``--skip_extract``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_kwargs):  # type: ignore
        return it


# ---------------------------------------------------------------------------
# ImageNet-1k wnid list
# ---------------------------------------------------------------------------

# Published alongside the Alibaba-MIIL ImageNet-21K preprocessing utilities.
_IN1K_WNIDS_URL = (
    "https://raw.githubusercontent.com/Alibaba-MIIL/ImageNet21K/main/"
    "dataset_preprocessing/processing_files/imagenet1k_wnids.txt"
)


def load_imagenet1k_wnids(cache_path: Path) -> Set[str]:
    """Return the set of 1000 ImageNet-1k wnids.

    Priority: local cache file -> download from the MIIL repo. If the download
    fails, the user is asked to place the file manually at ``cache_path``.
    """
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading ImageNet-1k wnid list -> {cache_path}")
        try:
            urllib.request.urlretrieve(_IN1K_WNIDS_URL, cache_path)
        except Exception as e:
            raise SystemExit(
                f"Failed to download ImageNet-1k wnid list: {e}\n"
                f"Please place the 1000-line wnid file manually at:\n"
                f"  {cache_path}\n"
                f"A suitable source is:\n  {_IN1K_WNIDS_URL}"
            )
    wnids = {ln.strip() for ln in cache_path.read_text().splitlines() if ln.strip()}
    if len(wnids) != 1000:
        raise SystemExit(
            f"Expected 1000 wnids in {cache_path}, got {len(wnids)}. "
            f"File may be corrupted; delete it and re-run."
        )
    return wnids


# ---------------------------------------------------------------------------
# Tarball path parsing
# ---------------------------------------------------------------------------

def _parse_member(name: str) -> Optional[Tuple[str, str, str]]:
    """Return ``(tar_split, wnid, filename)`` for a 21k-P image member, or
    ``None`` if the path doesn't match.

    Expected layout inside the tar::

        imagenet21k_resized/imagenet21k_{train,val}/<wnid>/<file>.JPEG
    """
    parts = name.replace("\\", "/").split("/")
    # Take the last three components defensively; the top-level dir name
    # varies slightly between MIIL releases.
    if len(parts) < 3:
        return None
    tar_split, wnid, fname = parts[-3], parts[-2], parts[-1]
    if tar_split not in ("imagenet21k_train", "imagenet21k_val"):
        return None
    if not (wnid.startswith("n") and len(wnid) == 9 and wnid[1:].isdigit()):
        return None
    if not fname:
        return None
    return tar_split, wnid, fname


# ---------------------------------------------------------------------------
# Phase 1: scan
# ---------------------------------------------------------------------------

def phase_scan(tar_path: Path, cache_path: Path) -> Dict[str, Dict[str, int]]:
    """Stream the tarball and count image files per (wnid, tar_split).

    Returns: ``{wnid: {"train": N_train, "val": N_val}}``.
    Result is cached to ``cache_path``; re-use on subsequent runs.
    """
    if cache_path.exists():
        print(f"[scan] cache found: {cache_path} (delete to re-scan)")
        return json.loads(cache_path.read_text())

    print(f"[scan] streaming {tar_path} (this is slow, ~15-45 min)...")
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0})
    n_seen = 0
    # 'r|gz' => non-seekable streaming read; memory constant.
    with tarfile.open(tar_path, "r|gz") as tar:
        for member in tqdm(tar, desc="scan", unit=" members"):
            if not member.isfile():
                continue
            parsed = _parse_member(member.name)
            if parsed is None:
                continue
            tar_split, wnid, _ = parsed
            key = "train" if tar_split == "imagenet21k_train" else "val"
            counts[wnid][key] += 1
            n_seen += 1

    counts = dict(counts)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(counts, indent=2, sort_keys=True))
    print(f"[scan] {len(counts)} classes, {n_seen} images. Cached -> {cache_path}")
    return counts


# ---------------------------------------------------------------------------
# Phase 2: select
# ---------------------------------------------------------------------------

def phase_select(
    counts: Dict[str, Dict[str, int]],
    in1k_wnids: Set[str],
    num_classes: int,
    min_samples: int,
    seed: int,
    meta_dir: Path,
) -> List[str]:
    """Pick ``num_classes`` wnids outside IN1k, stratified by wnid-offset
    bucket for coarse semantic diversity.

    Offset bucketing uses ``int(wnid[1:]) // 100_000`` — synsets with close
    offsets tend to cluster semantically in WordNet, so this is a rough (and
    dependency-free) proxy for category diversity. Round-robin over buckets
    yields 200 classes without any single bucket dominating.
    """
    all_wnids = sorted(counts.keys())
    candidates = [w for w in all_wnids if w not in in1k_wnids]
    candidates = [w for w in candidates if counts[w]["train"] >= min_samples]
    if len(candidates) < num_classes:
        raise SystemExit(
            f"Only {len(candidates)} candidates after filtering (need {num_classes}). "
            f"Try lowering --min_samples."
        )

    # Stratify: bucket by offset, shuffle inside each bucket, round-robin pick.
    rng = random.Random(seed)
    buckets: Dict[int, List[str]] = defaultdict(list)
    for w in candidates:
        buckets[int(w[1:]) // 100_000].append(w)
    for b in buckets.values():
        rng.shuffle(b)

    bucket_keys = sorted(buckets.keys())
    rng.shuffle(bucket_keys)

    selected: List[str] = []
    while len(selected) < num_classes:
        advanced = False
        for b in bucket_keys:
            if buckets[b]:
                selected.append(buckets[b].pop())
                advanced = True
                if len(selected) == num_classes:
                    break
        if not advanced:
            break  # buckets empty

    selected.sort()
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "selected_wnids.txt").write_text("\n".join(selected) + "\n")
    meta = {
        "num_classes": num_classes,
        "min_samples": min_samples,
        "seed": seed,
        "num_candidates_after_1k_exclude": len(
            [w for w in all_wnids if w not in in1k_wnids]
        ),
        "num_candidates_after_min_samples": len(candidates),
        "bucket_distribution": {
            str(b): sum(1 for w in selected if int(w[1:]) // 100_000 == b)
            for b in bucket_keys
        },
        "counts_selected": {w: counts[w] for w in selected},
    }
    (meta_dir / "selection_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[select] picked {len(selected)} classes across "
          f"{sum(1 for v in meta['bucket_distribution'].values() if v > 0)} buckets")
    print(f"[select] wrote {meta_dir / 'selected_wnids.txt'}")
    return selected


# ---------------------------------------------------------------------------
# Phase 3: extract
# ---------------------------------------------------------------------------

def _hashed_float(seed: int, *parts: str) -> float:
    """Deterministic float in [0, 1) derived from seed + path components."""
    h = hashlib.md5(f"{seed}|{'|'.join(parts)}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0x1_0000_0000  # 32-bit fraction


def phase_extract(
    tar_path: Path,
    selected_wnids: List[str],
    out_dir: Path,
    seed: int,
    val_ratio: float = 0.1,
) -> None:
    """Stream the tarball; write only selected classes into train/val/test.

    Split rules:
      - tar's imagenet21k_train  -> our_train (1 - val_ratio) + our_val (val_ratio)
      - tar's imagenet21k_val    -> our_test (kept as-is)
    """
    selected_set = set(selected_wnids)
    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    test_dir = out_dir / "test"
    for d in (train_dir, val_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    per_split_counts = {"train": 0, "val": 0, "test": 0}
    print(f"[extract] streaming {tar_path} -> {out_dir} "
          f"(writing only {len(selected_set)} classes)")
    with tarfile.open(tar_path, "r|gz") as tar:
        for member in tqdm(tar, desc="extract", unit=" members"):
            if not member.isfile():
                continue
            parsed = _parse_member(member.name)
            if parsed is None:
                continue
            tar_split, wnid, fname = parsed
            if wnid not in selected_set:
                continue

            if tar_split == "imagenet21k_val":
                dst_split = "test"
            else:
                # Deterministic per-file decision for train vs val.
                r = _hashed_float(seed, wnid, fname)
                dst_split = "val" if r < val_ratio else "train"

            dst_dir = out_dir / dst_split / wnid
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / fname
            if dst_path.exists():
                continue  # resumability

            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            with open(dst_path, "wb") as f:
                shutil.copyfileobj(extracted, f)
            per_split_counts[dst_split] += 1

    print(f"[extract] done. Wrote: {per_split_counts}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", required=True, type=Path,
                    help="Path to imagenet21k_resized.tar.gz.")
    ap.add_argument("--out_dir", required=True, type=Path,
                    help="Output root. Will contain train/, val/, test/, meta/.")
    ap.add_argument("--num_classes", type=int, default=200)
    ap.add_argument("--min_samples", type=int, default=500,
                    help="Minimum train images per class to be a candidate. "
                         "21k-P already pre-filters to >=500, so 500 keeps all.")
    ap.add_argument("--val_ratio", type=float, default=0.1,
                    help="Fraction of tar's train split diverted to our val.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--imagenet1k_wnids", type=Path, default=None,
                    help="Path to 1000-line IN1k wnid file. If missing, auto-downloaded.")
    ap.add_argument("--skip_scan", action="store_true")
    ap.add_argument("--skip_select", action="store_true")
    ap.add_argument("--skip_extract", action="store_true")
    args = ap.parse_args()

    if not args.tar.exists():
        sys.exit(f"Tar not found: {args.tar}")
    meta_dir = args.out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    in1k_cache = args.imagenet1k_wnids or (meta_dir / "imagenet1k_wnids.txt")
    in1k_wnids = load_imagenet1k_wnids(in1k_cache)

    scan_cache = meta_dir / "scan_cache.json"
    if args.skip_scan and not scan_cache.exists():
        sys.exit("--skip_scan given but scan_cache.json does not exist.")
    counts = (json.loads(scan_cache.read_text()) if args.skip_scan
              else phase_scan(args.tar, scan_cache))

    selected_path = meta_dir / "selected_wnids.txt"
    if args.skip_select:
        if not selected_path.exists():
            sys.exit("--skip_select given but selected_wnids.txt does not exist.")
        selected = [ln.strip() for ln in selected_path.read_text().splitlines() if ln.strip()]
        print(f"[select] reusing existing selection ({len(selected)} classes)")
    else:
        selected = phase_select(
            counts=counts,
            in1k_wnids=in1k_wnids,
            num_classes=args.num_classes,
            min_samples=args.min_samples,
            seed=args.seed,
            meta_dir=meta_dir,
        )

    if not args.skip_extract:
        phase_extract(
            tar_path=args.tar,
            selected_wnids=selected,
            out_dir=args.out_dir,
            seed=args.seed,
            val_ratio=args.val_ratio,
        )

    print("\nAll phases complete.")
    print(f"  meta   : {meta_dir}")
    print(f"  train  : {args.out_dir / 'train'}")
    print(f"  val    : {args.out_dir / 'val'}")
    print(f"  test   : {args.out_dir / 'test'}")


if __name__ == "__main__":
    main()
