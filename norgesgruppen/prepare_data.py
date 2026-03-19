"""
prepare_data.py - Download and convert NorgesGruppen shelf dataset to YOLO format.

Usage:
    python -m norgesgruppen.prepare_data \
        --data-url <URL> \
        --raw-dir data/raw \
        --yolo-dir data/yolo \
        --val-ratio 0.1

After running, data/yolo/ will contain:
    images/train/, images/val/
    labels/train/, labels/val/
    data.yaml
And norgesgruppen/category_mapping.json will be written (bundle this in the submission ZIP).
"""

import argparse
import json
import os
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import requests
from tqdm import tqdm

SECTIONS = ["Egg", "Frokost", "Knekkebrod", "Varmedrikker"]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(url: str, dest_path: Path) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest_path}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest_path.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    print(f"Extracting {archive_path} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(dest_dir)


# ---------------------------------------------------------------------------
# Category mapping
# ---------------------------------------------------------------------------

def build_category_mapping(raw_dir: Path):
    """
    Merge categories across all section annotation files.
    Returns:
        categories: sorted list of {"id": int, "name": str}
        coco_to_yolo: {coco_id: yolo_idx}
        yolo_to_coco: {yolo_idx: coco_id}
        names: list of category names in yolo_idx order
    """
    all_categories = {}  # coco_id -> name

    for section in SECTIONS:
        ann_path = _find_annotation_file(raw_dir, section)
        if ann_path is None:
            print(f"  WARNING: no annotation file found for section {section}")
            continue
        with open(ann_path) as f:
            data = json.load(f)
        for cat in data.get("categories", []):
            cid = cat["id"]
            if cid in all_categories and all_categories[cid] != cat["name"]:
                print(
                    f"  WARNING: category id {cid} has conflicting names: "
                    f"{all_categories[cid]!r} vs {cat['name']!r}"
                )
            all_categories[cid] = cat["name"]

    # Sort by coco id for deterministic ordering
    sorted_cats = sorted(all_categories.items(), key=lambda x: x[0])
    coco_to_yolo = {cid: idx for idx, (cid, _) in enumerate(sorted_cats)}
    yolo_to_coco = {idx: cid for idx, (cid, _) in enumerate(sorted_cats)}
    names = [name for _, name in sorted_cats]

    print(f"Found {len(sorted_cats)} categories across all sections.")
    return sorted_cats, coco_to_yolo, yolo_to_coco, names


def _find_annotation_file(raw_dir: Path, section: str) -> Path | None:
    candidates = [
        raw_dir / section / "annotations" / "annotations.json",
        raw_dir / section / "annotations.json",
        raw_dir / section / f"{section}_annotations.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Recursive search fallback
    for p in (raw_dir / section).rglob("*.json"):
        if "annotation" in p.name.lower():
            return p
    return None


# ---------------------------------------------------------------------------
# COCO → YOLO conversion
# ---------------------------------------------------------------------------

def convert_section(
    raw_dir: Path,
    section: str,
    coco_to_yolo: dict,
    images_out_dir: Path,
    labels_out_dir: Path,
) -> list[dict]:
    """
    Convert one section's COCO annotations to YOLO format.
    Returns list of image metadata dicts for split logic.
    """
    ann_path = _find_annotation_file(raw_dir, section)
    if ann_path is None:
        print(f"  SKIP: no annotation file for {section}")
        return []

    with open(ann_path) as f:
        coco = json.load(f)

    # Build lookup: image_id -> {file_name, width, height}
    img_info = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    ann_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        ann_by_image[ann["image_id"]].append(ann)

    # Find the images directory for this section
    img_src_dir = _find_images_dir(raw_dir, section)

    records = []
    skipped = 0
    for img_id, img_meta in img_info.items():
        fname = img_meta["file_name"]
        img_w = img_meta["width"]
        img_h = img_meta["height"]

        # Locate source image
        src_img = _find_image_file(img_src_dir, fname)
        if src_img is None:
            skipped += 1
            continue

        # Build unique image stem: section_imageId
        stem = f"{section}_{img_id}"

        # Build YOLO label lines
        lines = []
        for ann in ann_by_image[img_id]:
            coco_cat = ann["category_id"]
            if coco_cat not in coco_to_yolo:
                continue
            yolo_cls = coco_to_yolo[coco_cat]
            x, y, w, h = ann["bbox"]
            x_c = (x + w / 2) / img_w
            y_c = (y + h / 2) / img_h
            w_n = w / img_w
            h_n = h / img_h
            # Clamp to [0, 1]
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            w_n = max(0.001, min(1.0, w_n))
            h_n = max(0.001, min(1.0, h_n))
            lines.append(f"{yolo_cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

        records.append({
            "stem": stem,
            "src_img": src_img,
            "label_lines": lines,
            "section": section,
            "classes": list({ann["category_id"] for ann in ann_by_image[img_id]}),
        })

    if skipped:
        print(f"  WARNING: {skipped} images in {section} annotations not found on disk")
    print(f"  {section}: {len(records)} images ready")
    return records


def _find_images_dir(raw_dir: Path, section: str) -> Path:
    candidates = [raw_dir / section / "images", raw_dir / section]
    for c in candidates:
        if c.is_dir():
            return c
    return raw_dir / section


def _find_image_file(img_dir: Path, fname: str) -> Path | None:
    p = img_dir / fname
    if p.exists():
        return p
    # Search recursively
    for ext in ["jpg", "jpeg", "png", "JPG", "JPEG"]:
        matches = list(img_dir.rglob(f"{Path(fname).stem}.{ext}"))
        if matches:
            return matches[0]
    return None


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------

def split_records(
    records: list[dict],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Stratified split by section. Ensures all classes appear in train.
    """
    rng = random.Random(seed)

    # Group by section
    by_section: dict[str, list] = defaultdict(list)
    for r in records:
        by_section[r["section"]].append(r)

    train, val = [], []
    for section, imgs in by_section.items():
        shuffled = list(imgs)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_ratio))
        val.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])

    # Post-hoc: ensure every class seen in val also appears in train
    train_classes = set()
    for r in train:
        train_classes.update(r["classes"])

    rescued = []
    remaining_val = []
    for r in val:
        if any(c not in train_classes for c in r["classes"]):
            rescued.append(r)
            train_classes.update(r["classes"])
        else:
            remaining_val.append(r)

    if rescued:
        print(f"  Moved {len(rescued)} images from val→train to ensure class coverage")
        train.extend(rescued)
        val = remaining_val

    # Warn on rare classes
    class_counts: dict[int, int] = defaultdict(int)
    for r in train:
        for c in r["classes"]:
            class_counts[c] += 1
    rare = [c for c, cnt in class_counts.items() if cnt < 5]
    if rare:
        print(f"  WARNING: {len(rare)} classes have fewer than 5 training images")

    print(f"  Split: {len(train)} train / {len(val)} val")
    return train, val


# ---------------------------------------------------------------------------
# Write YOLO dataset
# ---------------------------------------------------------------------------

def write_split(
    records: list[dict],
    split: str,
    images_dir: Path,
    labels_dir: Path,
) -> None:
    img_out = images_dir / split
    lbl_out = labels_dir / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for r in tqdm(records, desc=f"Writing {split}"):
        stem = r["stem"]
        # Copy image
        dst_img = img_out / f"{stem}.jpg"
        src = Path(r["src_img"])
        if src.suffix.lower() in (".jpg", ".jpeg"):
            shutil.copy2(src, dst_img)
        else:
            # Convert to JPEG via OpenCV if needed
            import cv2
            img = cv2.imread(str(src))
            cv2.imwrite(str(dst_img), img)

        # Write label file
        lbl_path = lbl_out / f"{stem}.txt"
        with open(lbl_path, "w") as f:
            f.write("\n".join(r["label_lines"]))


def write_data_yaml(yolo_dir: Path, names: list[str]) -> None:
    yaml_path = yolo_dir / "data.yaml"
    abs_path = str(yolo_dir.resolve())
    lines = [
        f"path: {abs_path}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(names)}",
        "",
        "names:",
    ]
    for idx, name in enumerate(names):
        lines.append(f"  {idx}: {name}")
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {yaml_path}")


def write_category_mapping(
    repo_root: Path,
    coco_to_yolo: dict,
    yolo_to_coco: dict,
    names: list[str],
) -> None:
    mapping = {
        "coco_to_yolo": {str(k): v for k, v in coco_to_yolo.items()},
        "yolo_to_coco": {str(k): v for k, v in yolo_to_coco.items()},
        "names": names,
    }
    out_path = repo_root / "norgesgruppen" / "category_mapping.json"
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Wrote {out_path}  (bundle this in the submission ZIP)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare NorgesGruppen dataset for YOLOv8 training")
    parser.add_argument("--data-url", help="URL to download the dataset archive (zip)")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory to extract raw data into")
    parser.add_argument("--yolo-dir", default="data/yolo", help="Output directory for YOLO-format dataset")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of data to use for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    yolo_dir = Path(args.yolo_dir)
    repo_root = Path(__file__).parent.parent

    # 1. Download if URL provided
    if args.data_url:
        archive_path = raw_dir / "dataset.zip"
        if not archive_path.exists():
            download_file(args.data_url, archive_path)
        extract_archive(archive_path, raw_dir)
    else:
        print(f"No --data-url provided; assuming data already in {raw_dir}")

    # 2. Build category mapping
    print("\nBuilding category mapping...")
    sorted_cats, coco_to_yolo, yolo_to_coco, names = build_category_mapping(raw_dir)

    # 3. Convert all sections
    print("\nConverting COCO annotations to YOLO format...")
    all_records = []
    for section in SECTIONS:
        records = convert_section(
            raw_dir, section, coco_to_yolo,
            yolo_dir / "images", yolo_dir / "labels",
        )
        all_records.extend(records)

    print(f"\nTotal images: {len(all_records)}")

    # 4. Split
    print("\nSplitting dataset...")
    train_records, val_records = split_records(all_records, val_ratio=args.val_ratio, seed=args.seed)

    # 5. Write YOLO dataset
    print("\nWriting YOLO dataset...")
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"
    write_split(train_records, "train", images_dir, labels_dir)
    write_split(val_records, "val", images_dir, labels_dir)

    # 6. Write data.yaml
    write_data_yaml(yolo_dir, names)

    # 7. Write category mapping (for bundling in ZIP)
    write_category_mapping(repo_root, coco_to_yolo, yolo_to_coco, names)

    print("\nDone! Next steps:")
    print(f"  1. Upload {yolo_dir}/ to your remote GPU machine")
    print("  2. Run: python -m norgesgruppen.train --data <yolo_dir>/data.yaml")


if __name__ == "__main__":
    main()
