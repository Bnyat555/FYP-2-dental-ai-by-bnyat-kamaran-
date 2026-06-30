from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(r"C:\Users\MSI\Desktop\Dental-Ai\dental_ai")
SEG_ROOT = PROJECT_ROOT / "data" / "segment_teeth"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "seg_dataset_check"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def list_images(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(files)


def list_labels(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.txt"))


def parse_label_file(label_path: Path) -> tuple[list[np.ndarray], int, int]:
    polygons = []
    bad_lines = 0
    bad_polygons = 0

    raw_lines = label_path.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            bad_lines += 1
            continue

        try:
            values = [float(x) for x in parts]
        except ValueError:
            bad_lines += 1
            continue

        cls = int(round(values[0]))
        coords = values[1:]

        if cls != 0:
            bad_polygons += 1
            continue

        if len(coords) < 6 or len(coords) % 2 != 0:
            bad_lines += 1
            continue

        pts = np.array(coords, dtype=np.float32).reshape(-1, 2)

        if np.any(pts < 0.0) or np.any(pts > 1.0):
            bad_polygons += 1
            continue

        area = polygon_area(pts)
        if area <= 1e-6:
            bad_polygons += 1
            continue

        polygons.append(pts)

    return polygons, bad_lines, bad_polygons


def draw_polygons(image: np.ndarray, polygons_norm: list[np.ndarray]) -> np.ndarray:
    h, w = image.shape[:2]
    canvas = image.copy()

    for idx, poly_norm in enumerate(polygons_norm, start=1):
        poly_px = poly_norm.copy()
        poly_px[:, 0] *= w
        poly_px[:, 1] *= h
        poly_px = np.round(poly_px).astype(np.int32)

        if len(poly_px) >= 3:
            cv2.polylines(canvas, [poly_px], isClosed=True, color=(0, 255, 0), thickness=2)
            cx = int(np.mean(poly_px[:, 0]))
            cy = int(np.mean(poly_px[:, 1]))
            cv2.putText(
                canvas,
                str(idx),
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return canvas


def verify_split(split: str, render_count: int) -> None:
    images_dir = SEG_ROOT / "images" / split
    labels_dir = SEG_ROOT / "labels" / split

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images folder: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels folder: {labels_dir}")

    images = list_images(images_dir)
    labels = list_labels(labels_dir)

    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_labels = sorted(image_stems - label_stems)
    orphan_labels = sorted(label_stems - image_stems)

    total_objects = 0
    bad_lines_total = 0
    bad_polygons_total = 0
    empty_labels = 0
    min_objects = 10**9
    max_objects = 0

    valid_render_items: list[tuple[Path, list[np.ndarray]]] = []

    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        polygons, bad_lines, bad_polygons = parse_label_file(label_path)
        bad_lines_total += bad_lines
        bad_polygons_total += bad_polygons

        if len(polygons) == 0:
            empty_labels += 1
            continue

        total_objects += len(polygons)
        min_objects = min(min_objects, len(polygons))
        max_objects = max(max_objects, len(polygons))
        valid_render_items.append((image_path, polygons))

    if min_objects == 10**9:
        min_objects = 0

    print("=" * 80)
    print(f"Split: {split}")
    print(f"Images found: {len(images)}")
    print(f"Label files found: {len(labels)}")
    print(f"Images missing labels: {len(missing_labels)}")
    print(f"Orphan label files: {len(orphan_labels)}")
    print(f"Empty label files: {empty_labels}")
    print(f"Bad label lines: {bad_lines_total}")
    print(f"Bad polygons: {bad_polygons_total}")
    print(f"Total valid objects: {total_objects}")

    if len(images) > 0:
        avg_objects = total_objects / len(images)
        print(f"Average objects per image: {avg_objects:.2f}")
        print(f"Min objects per image: {min_objects}")
        print(f"Max objects per image: {max_objects}")

    if missing_labels:
        print("\nFirst missing labels:")
        for name in missing_labels[:10]:
            print(f"  {name}")

    if orphan_labels:
        print("\nFirst orphan labels:")
        for name in orphan_labels[:10]:
            print(f"  {name}")

    if render_count > 0 and valid_render_items:
        out_dir = OUTPUT_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)

        sample_items = random.sample(valid_render_items, k=min(render_count, len(valid_render_items)))
        for image_path, polygons in sample_items:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            rendered = draw_polygons(image, polygons)
            save_path = out_dir / f"{image_path.stem}_check.png"
            cv2.imwrite(str(save_path), rendered)

        print(f"\nRendered samples saved to: {out_dir}")

    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--render", type=int, default=12)
    args = parser.parse_args()

    if args.split in {"train", "all"}:
        verify_split("train", args.render)

    if args.split in {"val", "all"}:
        verify_split("val", args.render)


if __name__ == "__main__":
    main()