"""
train.py - Fine-tune YOLOv8 on the NorgesGruppen shelf dataset.

Run on a machine with a GPU (NVIDIA L4 recommended).

Usage:
    python -m norgesgruppen.train \
        --data data/yolo/data.yaml \
        --model yolov8m.pt \
        --epochs 100 \
        --imgsz 640 \
        --project runs/detect \
        --name norgesgruppen

After training, the best weights are at:
    runs/detect/norgesgruppen/weights/best.pt
"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on NorgesGruppen dataset")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8m.pt", help="YOLOv8 model variant (e.g. yolov8m.pt, yolov8l.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size (640 or 1280)")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size (-1 = auto)")
    parser.add_argument("--project", default="runs/detect", help="Output project directory")
    parser.add_argument("--name", default="norgesgruppen", help="Run name")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--export-onnx", action="store_true", help="Export best model to ONNX after training")
    return parser.parse_args()


def main():
    args = parse_args()

    # Import here so the module is importable even without ultralytics installed
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"data.yaml not found at {data_path}. "
            "Run prepare_data.py first."
        )

    print(f"Loading model: {args.model}")
    if args.resume:
        # Resume from last checkpoint in the project/name directory
        last_ckpt = Path(args.project) / args.name / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"Cannot resume: {last_ckpt} not found")
        model = YOLO(str(last_ckpt))
    else:
        model = YOLO(args.model)

    print(f"Training on: {args.data}")
    print(f"Epochs: {args.epochs}, imgsz: {args.imgsz}, batch: {args.batch}")

    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        resume=args.resume,

        # Optimizer / LR
        lr0=args.lr0,
        lrf=0.01,           # final LR = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,

        # Training
        patience=20,        # early stopping patience
        save=True,
        save_period=10,     # checkpoint every N epochs
        val=True,
        plots=True,
        workers=8,

        # Augmentation — tuned for small shelf dataset
        mosaic=1.0,         # mix 4 images; key for small dataset
        copy_paste=0.3,     # paste objects across images; helps rare classes
        close_mosaic=10,    # disable mosaic in last 10 epochs for stability
        degrees=5.0,        # small rotation
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0002,
        flipud=0.0,         # shelves are never upside down
        fliplr=0.5,
        mixup=0.0,          # harmful for dense detection
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )

    # Report val metrics
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_weights}")

    # Final validation with per-class AP
    print("\nRunning final validation...")
    best_model = YOLO(str(best_weights))
    val_results = best_model.val(data=str(data_path), imgsz=args.imgsz, verbose=True)
    print(f"  mAP50:   {val_results.box.map50:.4f}")
    print(f"  mAP50-95: {val_results.box.map:.4f}")

    # Optional ONNX export
    if args.export_onnx:
        print("\nExporting to ONNX...")
        best_model.export(format="onnx", dynamic=True, simplify=True, imgsz=args.imgsz)
        onnx_path = best_weights.with_suffix(".onnx")
        print(f"  Exported to {onnx_path}")

    print(f"\nNext: python -m norgesgruppen.package --weights {best_weights}")


if __name__ == "__main__":
    main()
