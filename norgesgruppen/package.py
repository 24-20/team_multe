"""
package.py - Build the submission ZIP for the NorgesGruppen challenge.

The ZIP must have run.py at the root (not inside a subfolder).
Maximum total size: 420 MB.

Usage:
    python -m norgesgruppen.package \
        --weights runs/detect/norgesgruppen/weights/best.pt \
        --output submission.zip
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

MAX_SIZE_BYTES = 420 * 1024 * 1024  # 420 MB


def build_file_list(weights_path: Path, repo_root: Path) -> list[tuple[Path, str]]:
    """
    Returns list of (source_path, archive_name) tuples.
    All archive_names are flat (no subdirectory prefix).
    """
    run_py = repo_root / "norgesgruppen" / "run.py"
    mapping_json = repo_root / "norgesgruppen" / "category_mapping.json"

    files = [
        (run_py, "run.py"),
        (weights_path, "best.pt"),
        (mapping_json, "category_mapping.json"),
    ]
    return files


def validate_files(files: list[tuple[Path, str]]) -> None:
    missing = [src for src, _ in files if not src.exists()]
    if missing:
        for p in missing:
            print(f"  MISSING: {p}")
        raise FileNotFoundError(
            f"{len(missing)} required file(s) not found. "
            "Run prepare_data.py and train.py first."
        )


def check_size(files: list[tuple[Path, str]]) -> int:
    total = sum(src.stat().st_size for src, _ in files)
    total_mb = total / 1024 / 1024
    print(f"Total uncompressed size: {total_mb:.1f} MB")
    if total > MAX_SIZE_BYTES:
        raise ValueError(
            f"Files total {total_mb:.1f} MB, exceeds the {MAX_SIZE_BYTES // 1024 // 1024} MB limit. "
            "Consider using a smaller model variant."
        )
    return total


def create_zip(files: list[tuple[Path, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for src, arc_name in files:
            print(f"  Adding {arc_name} ({src.stat().st_size / 1024 / 1024:.1f} MB)")
            zf.write(src, arc_name)
    zip_size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"ZIP size: {zip_size_mb:.1f} MB  →  {output_path}")


def verify_zip(output_path: Path) -> None:
    print("\nVerifying ZIP contents:")
    with zipfile.ZipFile(output_path, "r") as zf:
        names = zf.namelist()
        for name in names:
            info = zf.getinfo(name)
            print(f"  {name:<40} {info.file_size / 1024:.0f} KB")

    # run.py must be at root (no path separator)
    if "run.py" not in names:
        raise ValueError("run.py not found in ZIP — submission will be rejected")
    if "/" in "run.py" or "\\" in "run.py":
        raise ValueError("run.py is inside a subdirectory — it must be at ZIP root")

    if "best.pt" not in names:
        print("  WARNING: best.pt not found in ZIP — model weights missing")
    if "category_mapping.json" not in names:
        print("  WARNING: category_mapping.json not found — inference will fail")

    print("\nVerification passed.")


def main():
    parser = argparse.ArgumentParser(description="Package NorgesGruppen submission ZIP")
    parser.add_argument(
        "--weights",
        default="runs/detect/norgesgruppen/weights/best.pt",
        help="Path to trained model weights (best.pt)",
    )
    parser.add_argument(
        "--output",
        default="submission.zip",
        help="Output ZIP file path",
    )
    args = parser.parse_args()

    weights_path = Path(args.weights)
    output_path = Path(args.output)
    repo_root = Path(__file__).parent.parent

    print(f"Building submission ZIP: {output_path}")
    print(f"  Weights: {weights_path}")

    files = build_file_list(weights_path, repo_root)

    print("\nChecking files...")
    validate_files(files)

    print("\nChecking size...")
    check_size(files)

    print("\nCreating ZIP...")
    create_zip(files, output_path)

    verify_zip(output_path)

    print(f"\nReady to submit: {output_path.resolve()}")
    print("Upload at: https://app.ainm.no/submit/norgesgruppen-data")


if __name__ == "__main__":
    main()
