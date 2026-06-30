# src/prepare_identity_dataset.py
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"C:\Users\MSI\Desktop\FYP AI Data Set")

TRAIN_JSON = SOURCE_ROOT / "training_data" / "quadrant_enumeration" / "train_quadrant_enumeration.json"
TRAIN_IMAGES = SOURCE_ROOT / "training_data" / "quadrant_enumeration" / "xrays"

VAL_JSON = SOURCE_ROOT / "validation_triple.json"
VAL_IMAGES = SOURCE_ROOT / "validation_data" / "quadrant_enumeration_disease" / "xrays"

IDENTITY_ROOT = PROJECT_ROOT / "data" / "tooth_identity_32"
META_ROOT = PROJECT_ROOT / "data" / "metadata_identity"

QUADRANT_MAP = {
    "1": "UR",
    "2": "UL",
    "3": "LL",
    "4": "LR",
    1: "UR",
    2: "UL",
    3: "LL",
    4: "LR",
}

TOOTH_MAP = {
    "1": "CI",
    "2": "LI",
    "3": "C",
    "4": "P1",
    "5": "P2",
    "6": "M1",
    "7": "M2",
    "8": "M3",
    1: "CI",
    2: "LI",
    3: "C",
    4: "P1",
    5: "P2",
    6: "M1",
    7: "M2",
    8: "M3",
}


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_name_map(data: dict, key: str) -> dict:
    mapping = {}
    for item in data.get(key, []):
        mapping[item["id"]] = str(item["name"])
    return mapping


def make_all_class_dirs():
    reset_dir(IDENTITY_ROOT / "train")
    reset_dir(IDENTITY_ROOT / "val")
    reset_dir(IDENTITY_ROOT / "test")
    reset_dir(META_ROOT)

    quadrants = ["UR", "UL", "LR", "LL"]
    teeth = ["CI", "LI", "C", "P1", "P2", "M1", "M2", "M3"]

    for split in ["train", "val", "test"]:
        for q in quadrants:
            for t in teeth:
                (IDENTITY_ROOT / split / f"{q}-{t}").mkdir(parents=True, exist_ok=True)


def segmentation_to_mask(segmentation, img_h: int, img_w: int) -> np.ndarray:
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    if not segmentation:
        return mask

    if isinstance(segmentation, list):
        for poly in segmentation:
            if not poly or len(poly) < 6:
                continue
            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)
            cv2.fillPoly(mask, [pts], 255)

    return mask


def process_split(split: str, json_path: Path, image_dir: Path) -> None:
    data = load_json(json_path)

    images_by_id = {img["id"]: img for img in data["images"]}
    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    category_1_map = build_name_map(data, "categories_1")
    category_2_map = build_name_map(data, "categories_2")

    rows = []
    written = 0

    for image_id, image_info in images_by_id.items():
        image_path = image_dir / image_info["file_name"]
        if not image_path.exists():
            print(f"[{split}] Missing image: {image_path}")
            continue

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[{split}] Failed to read image: {image_path}")
            continue

        img_h, img_w = img.shape[:2]

        for ann in anns_by_image.get(image_id, []):
            bbox = ann.get("bbox", [])
            segmentation = ann.get("segmentation", [])

            if len(bbox) != 4:
                continue

            x, y, w, h = map(float, bbox)
            if w <= 5 or h <= 5:
                continue

            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(img_w, int(round(x + w)))
            y2 = min(img_h, int(round(y + h)))

            if x2 <= x1 or y2 <= y1:
                continue

            full_mask = segmentation_to_mask(segmentation, img_h, img_w)
            crop_img = img[y1:y2, x1:x2]
            crop_mask = full_mask[y1:y2, x1:x2]

            if crop_img.size == 0 or crop_mask.size == 0:
                continue

            if cv2.countNonZero(crop_mask) == 0:
                continue

            masked_crop = cv2.bitwise_and(crop_img, crop_img, mask=crop_mask)

            quadrant_raw = category_1_map.get(ann.get("category_id_1"))
            tooth_raw = category_2_map.get(ann.get("category_id_2"))

            quadrant = QUADRANT_MAP.get(quadrant_raw)
            tooth_type = TOOTH_MAP.get(tooth_raw)

            if quadrant is None or tooth_type is None:
                continue

            class_name = f"{quadrant}-{tooth_type}"
            out_name = f"{Path(image_info['file_name']).stem}_ann_{ann['id']}_{class_name}.png"
            out_path = IDENTITY_ROOT / split / class_name / out_name

            cv2.imwrite(str(out_path), masked_crop)
            written += 1

            rows.append({
                "split": split,
                "source_image": image_info["file_name"],
                "annotation_id": ann["id"],
                "class_name": class_name,
                "quadrant": quadrant,
                "tooth_type": tooth_type,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "output_file": out_name,
            })

    csv_path = META_ROOT / f"{split}_identity_32.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "source_image",
                "annotation_id",
                "class_name",
                "quadrant",
                "tooth_type",
                "x1",
                "y1",
                "x2",
                "y2",
                "output_file",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[{split}] Identity crops written: {written}")
    print(f"[{split}] Metadata saved: {csv_path}")


def main():
    make_all_class_dirs()

    process_split(
        split="train",
        json_path=TRAIN_JSON,
        image_dir=TRAIN_IMAGES,
    )

    process_split(
        split="val",
        json_path=VAL_JSON,
        image_dir=VAL_IMAGES,
    )

    print("Done.")


if __name__ == "__main__":
    main()