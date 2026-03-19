"""
run.py - Submission entrypoint for the NorgesGruppen object detection challenge.

This file (and best.pt + category_mapping.json) are placed at the ROOT of the submission ZIP.

Sandbox environment (pre-installed):
    PyTorch, ultralytics (YOLOv8), ONNX Runtime, OpenCV
    GPU: NVIDIA L4 (24 GB VRAM)

Contract:
    Input:  JPEG images at /data/images/img_XXXXX.jpg
    Output: JSON array printed to stdout, each entry:
            {
                "bbox":        [x, y, width, height],  # top-left corner, pixels
                "category_id": int,                     # original COCO category_id
                "confidence":  float,
                "image_id":    str                      # filename stem, e.g. "img_00001"
            }

Timeout: 300 seconds total for all images.
"""

import glob
import json
import os
import sys

import torch
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

IMAGE_DIR = "/data/images"
BATCH_SIZE = 8
CONF_THRESHOLD = 0.1    # low threshold maximises recall for mAP evaluation
IOU_THRESHOLD = 0.45    # NMS IoU threshold
IMG_SIZE = 640
USE_TTA = True          # test-time augmentation (+1-3% mAP, well within 300s budget)

# ---------------------------------------------------------------------------
# Locate files relative to this script (all at ZIP root)
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "best.pt")
MAPPING_PATH = os.path.join(SCRIPT_DIR, "category_mapping.json")


def load_mapping() -> dict:
    """Load yolo_idx -> coco_category_id mapping from bundled JSON."""
    with open(MAPPING_PATH) as f:
        data = json.load(f)
    # Keys are strings in JSON; convert to int
    return {int(k): int(v) for k, v in data["yolo_to_coco"].items()}


def discover_images() -> list[str]:
    paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")))
    return paths


def run_inference(model: YOLO, image_paths: list[str], yolo_to_coco: dict) -> list[dict]:
    predictions = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i in range(0, len(image_paths), BATCH_SIZE):
        batch = image_paths[i : i + BATCH_SIZE]
        results = model.predict(
            batch,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMG_SIZE,
            device=device,
            augment=USE_TTA,
            verbose=False,
        )
        for path, result in zip(batch, results):
            image_id = os.path.splitext(os.path.basename(path))[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for j in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[j].tolist()
                w = x2 - x1
                h = y2 - y1
                yolo_cls = int(boxes.cls[j].item())
                conf = float(boxes.conf[j].item())
                coco_cat_id = yolo_to_coco.get(yolo_cls, yolo_cls)
                predictions.append({
                    "bbox": [x1, y1, w, h],
                    "category_id": coco_cat_id,
                    "confidence": conf,
                    "image_id": image_id,
                })

    return predictions


def main():
    # Load category mapping
    if not os.path.exists(MAPPING_PATH):
        print(f"ERROR: category_mapping.json not found at {MAPPING_PATH}", file=sys.stderr)
        sys.exit(1)
    yolo_to_coco = load_mapping()

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: model not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)
    model = YOLO(MODEL_PATH)

    # Discover images
    image_paths = discover_images()
    if not image_paths:
        print(f"WARNING: no images found in {IMAGE_DIR}", file=sys.stderr)
        print(json.dumps([]))
        return

    # Run inference
    predictions = run_inference(model, image_paths, yolo_to_coco)

    # Output JSON
    print(json.dumps(predictions))


if __name__ == "__main__":
    main()
