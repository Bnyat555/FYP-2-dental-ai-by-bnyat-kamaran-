from __future__ import annotations

from pathlib import Path
import json
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"C:\Users\MSI\Desktop\FYP AI Data Set")

TRAIN_JSON = SOURCE_ROOT / "training_data" / "quadrant_enumeration" / "train_quadrant_enumeration.json"
TRAIN_IMAGES = SOURCE_ROOT / "training_data" / "quadrant_enumeration" / "xrays"

SEG_ROOT = PROJECT_ROOT / "data" / "segment_teeth"
SPLIT_INFO_PATH = SEG_ROOT / "split_info.json"

SEED = 42
VAL_RATIO = 0.15


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def polygon_area(poly: list[float]) -> float:
    if len(poly) < 6:
        return 0.0

    pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def bbox_to_polygon(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def choose_polygon(segmentation, bbox: list[float]) -> list[float]:
    if isinstance(segmentation, list) and len(segmentation) > 0:
        valid = [poly for poly in segmentation if isinstance(poly, list) and len(poly) >= 6]
        if valid:
            valid.sort(key=polygon_area, reverse=True)
            return valid[0]
    return bbox_to_polygon(bbox)


def to_yolo_seg_line(poly: list[float], img_w: float, img_h: float) -> str:
    coords = []
    for i in range(0, len(poly), 2):
        x = max(0.0, min(float(poly[i]) / img_w, 1.0))
        y = max(0.0, min(float(poly[i + 1]) / img_h, 1.0))
        coords.append(f"{x:.6f}")
        coords.append(f"{y:.6f}")
    return "0 " + " ".join(coords)


def build_labeled_items(json_path: Path, image_dir: Path) -> list[dict]:
    data = load_json(json_path)

    images_by_id = {img["id"]: img for img in data["images"]}
    anns_by_image: dict[int, list[dict]] = {}

    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    items = []
    missing_images = 0
    skipped_no_labels = 0

    for image_id, image_info in images_by_id.items():
        src_img = image_dir / image_info["file_name"]
        if not src_img.exists():
            missing_images += 1
            continue

        img_w = float(image_info["width"])
        img_h = float(image_info["height"])

        label_lines = []
        for ann in anns_by_image.get(image_id, []):
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue

            x, y, w, h = map(float, bbox)
            if w <= 2 or h <= 2:
                continue

            poly = choose_polygon(ann.get("segmentation", []), bbox)
            if len(poly) < 6:
                continue

            label_lines.append(to_yolo_seg_line(poly, img_w, img_h))

        if not label_lines:
            skipped_no_labels += 1
            continue

        items.append(
            {
                "image_id": image_id,
                "file_name": image_info["file_name"],
                "src_img": src_img,
                "label_lines": label_lines,
                "num_objects": len(label_lines),
                "width": img_w,
                "height": img_h,
            }
        )

    print(f"Total labeled items from source: {len(items)}")
    print(f"Missing image files: {missing_images}")
    print(f"Images skipped because no valid labels were found: {skipped_no_labels}")

    return items


def split_items(items: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not items:
        raise ValueError("No labeled items found. Dataset build failed.")

    items = sorted(items, key=lambda x: x["file_name"])
    rng = random.Random(seed)
    rng.shuffle(items)

    n_total = len(items)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_val = min(n_val, n_total - 1)

    val_items = sorted(items[:n_val], key=lambda x: x["file_name"])
    train_items = sorted(items[n_val:], key=lambda x: x["file_name"])

    return train_items, val_items


def write_split(split_name: str, items: list[dict]) -> None:
    out_img_dir = SEG_ROOT / "images" / split_name
    out_lbl_dir = SEG_ROOT / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    written = 0
    total_objects = 0

    for item in items:
        dst_img = out_img_dir / item["file_name"]
        shutil.copy2(item["src_img"], dst_img)
        copied += 1

        lbl_path = out_lbl_dir / f"{Path(item['file_name']).stem}.txt"
        lbl_path.write_text("\n".join(item["label_lines"]), encoding="utf-8")
        written += 1
        total_objects += item["num_objects"]

    avg_objects = total_objects / len(items) if items else 0.0

    print(f"[{split_name}] images copied: {copied}")
    print(f"[{split_name}] label files written: {written}")
    print(f"[{split_name}] total tooth instances: {total_objects}")
    print(f"[{split_name}] average tooth instances per image: {avg_objects:.2f}")


def save_split_info(train_items: list[dict], val_items: list[dict], seed: int, val_ratio: float) -> None:
    payload = {
        "source_json": str(TRAIN_JSON),
        "source_images": str(TRAIN_IMAGES),
        "seed": seed,
        "val_ratio": val_ratio,
        "train_count": len(train_items),
        "val_count": len(val_items),
        "train_files": [item["file_name"] for item in train_items],
        "val_files": [item["file_name"] for item in val_items],
    }
    SPLIT_INFO_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Split info saved to: {SPLIT_INFO_PATH}")


def main() -> None:
    reset_dir(SEG_ROOT / "images" / "train")
    reset_dir(SEG_ROOT / "images" / "val")
    reset_dir(SEG_ROOT / "labels" / "train")
    reset_dir(SEG_ROOT / "labels" / "val")

    items = build_labeled_items(TRAIN_JSON, TRAIN_IMAGES)
    train_items, val_items = split_items(items, val_ratio=VAL_RATIO, seed=SEED)

    write_split("train", train_items)
    write_split("val", val_items)
    save_split_info(train_items, val_items, seed=SEED, val_ratio=VAL_RATIO)

    print("Segmentation dataset ready.")
    print("Note: validation_triple.json is not used for segmentation train or val.")


if __name__ == "__main__":
    main()