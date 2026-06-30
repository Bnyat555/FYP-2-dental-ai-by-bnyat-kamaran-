import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"C:\Users\MSI\Desktop\FYP AI Data Set")

TRAIN_JSON = SOURCE_ROOT / "training_data" / "quadrant_enumeration" / "train_quadrant_enumeration.json"
TRAIN_IMAGES = SOURCE_ROOT / "training_data" / "quadrant_enumeration" / "xrays"

VAL_JSON = SOURCE_ROOT / "validation_triple.json"
VAL_IMAGES = SOURCE_ROOT / "validation_data" / "quadrant_enumeration_disease" / "xrays"

DETECT_ROOT = PROJECT_ROOT / "data" / "detect_teeth"
CLS_ROOT = PROJECT_ROOT / "data" / "quadrant_cls"
META_ROOT = PROJECT_ROOT / "data" / "metadata"

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


def bbox_to_yolo(x: float, y: float, w: float, h: float, img_w: int, img_h: int):
    x = max(0.0, x)
    y = max(0.0, y)
    w = max(0.0, w)
    h = max(0.0, h)

    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h

    return cx, cy, nw, nh


def prepare_output_folders() -> None:
    reset_dir(DETECT_ROOT / "images" / "train")
    reset_dir(DETECT_ROOT / "images" / "val")
    reset_dir(DETECT_ROOT / "labels" / "train")
    reset_dir(DETECT_ROOT / "labels" / "val")

    reset_dir(CLS_ROOT / "train")
    reset_dir(CLS_ROOT / "val")
    reset_dir(CLS_ROOT / "test")

    for split in ["train", "val", "test"]:
        for quadrant in ["UR", "UL", "LL", "LR"]:
            (CLS_ROOT / split / quadrant).mkdir(parents=True, exist_ok=True)

    reset_dir(META_ROOT)


def process_split(split: str, json_path: Path, image_dir: Path) -> None:
    data = load_json(json_path)

    images_by_id = {img["id"]: img for img in data["images"]}
    ann_by_image = defaultdict(list)

    for ann in data["annotations"]:
        ann_by_image[ann["image_id"]].append(ann)

    category_1_map = build_name_map(data, "categories_1")
    category_2_map = build_name_map(data, "categories_2")

    rows = []
    copied_images = 0
    written_label_files = 0
    written_crops = 0

    for image_id, image_info in images_by_id.items():
        source_image = image_dir / image_info["file_name"]

        if not source_image.exists():
            print(f"[{split}] Missing image: {source_image}")
            continue

        target_image = DETECT_ROOT / "images" / split / image_info["file_name"]
        shutil.copy2(source_image, target_image)
        copied_images += 1

        img_w = int(image_info["width"])
        img_h = int(image_info["height"])

        label_lines = []
        full_img = None

        for ann in ann_by_image.get(image_id, []):
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue

            x, y, w, h = map(float, bbox)

            if w <= 1 or h <= 1:
                continue

            cx, cy, nw, nh = bbox_to_yolo(x, y, w, h, img_w, img_h)

            if nw <= 0 or nh <= 0:
                continue

            label_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

            if full_img is None:
                full_img = cv2.imread(str(source_image), cv2.IMREAD_UNCHANGED)
                if full_img is None:
                    print(f"[{split}] Failed to read image: {source_image}")
                    break

            x1 = max(0, int(round(x)))
            y1 = max(0, int(round(y)))
            x2 = min(full_img.shape[1], int(round(x + w)))
            y2 = min(full_img.shape[0], int(round(y + h)))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = full_img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            quadrant_raw = category_1_map.get(ann.get("category_id_1"), str(ann.get("category_id_1", "")))
            tooth_raw = category_2_map.get(ann.get("category_id_2"), str(ann.get("category_id_2", "")))

            quadrant = QUADRANT_MAP.get(quadrant_raw, f"Q{quadrant_raw}")
            tooth_type = TOOTH_MAP.get(tooth_raw, str(tooth_raw))

            crop_name = f"{Path(image_info['file_name']).stem}_tooth_{ann['id']}_{quadrant}_{tooth_type}.png"
            crop_path = CLS_ROOT / split / quadrant / crop_name
            cv2.imwrite(str(crop_path), crop)
            written_crops += 1

            rows.append({
                "split": split,
                "source_image": image_info["file_name"],
                "annotation_id": ann["id"],
                "quadrant_id": ann.get("category_id_1"),
                "quadrant_label": quadrant,
                "tooth_enum_id": ann.get("category_id_2"),
                "tooth_type": tooth_type,
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "crop_file": crop_name,
            })

        label_path = DETECT_ROOT / "labels" / split / f"{Path(image_info['file_name']).stem}.txt"
        label_path.write_text("\n".join(label_lines), encoding="utf-8")
        written_label_files += 1

    meta_path = META_ROOT / f"{split}_tooth_metadata.csv"
    fieldnames = [
        "split",
        "source_image",
        "annotation_id",
        "quadrant_id",
        "quadrant_label",
        "tooth_enum_id",
        "tooth_type",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
        "crop_file",
    ]

    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("")
    print(f"[{split.upper()}]")
    print(f"Images copied: {copied_images}")
    print(f"Label files written: {written_label_files}")
    print(f"Tooth crops written: {written_crops}")
    print(f"Metadata saved: {meta_path}")


def main() -> None:
    prepare_output_folders()

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

    print("")
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
