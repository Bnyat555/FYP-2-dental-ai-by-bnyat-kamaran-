import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(r"C:\Users\MSI\Desktop\Dental-Ai\dental_ai")

RAW_JSON_PATH = Path(
    r"C:\Users\MSI\Desktop\FYP AI Data Set\training_data\quadrant-enumeration-disease\train_quadrant_enumeration_disease.json"
)

RAW_IMAGES_DIR = Path(
    r"C:\Users\MSI\Desktop\Dental-Ai\dental_ai\data\pathology_raw_images\quadrant-enumeration-disease\xrays"
)

YOLO_BASE_DIR = PROJECT_ROOT / "data" / "pathology"
IMAGES_TRAIN = YOLO_BASE_DIR / "images" / "train"
LABELS_TRAIN = YOLO_BASE_DIR / "labels" / "train"

# Final YOLO classes
# 0 = impacted
# 1 = caries
# 2 = periapical_lesion


def setup_directories():
    print("Cleaning and setting up YOLO directories...")
    if YOLO_BASE_DIR.exists():
        shutil.rmtree(YOLO_BASE_DIR)

    IMAGES_TRAIN.mkdir(parents=True, exist_ok=True)
    LABELS_TRAIN.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_text(text):
    return str(text).strip().lower().replace("_", " ").replace("-", " ")


def build_id_to_name(category_list):
    lookup = {}
    for item in category_list:
        cat_id = item.get("id")
        name = item.get("name", "")
        if cat_id is not None:
            lookup[cat_id] = normalize_text(name)
    return lookup


def find_category3_lookup(data):
    preferred_keys = ["categories_3", "category_3", "diseases", "disease_categories"]

    for key in preferred_keys:
        value = data.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            print(f"Using category table: {key}")
            return build_id_to_name(value)

    for key, value in data.items():
        if key.lower().startswith("categor") and isinstance(value, list) and value and isinstance(value[0], dict):
            if key.lower() == "categories":
                continue
            print(f"Using category table: {key}")
            return build_id_to_name(value)

    return {}


def build_disease_map_from_category3_lookup(category3_lookup):
    disease_map = {}

    for cat_id, name in category3_lookup.items():
        n = normalize_text(name)

        if "impacted" in n:
            disease_map[cat_id] = 0
        elif "caries" in n:
            disease_map[cat_id] = 1
        elif "periapical" in n:
            disease_map[cat_id] = 2

    return disease_map


def coco_bbox_to_yolo(bbox, img_w, img_h):
    if not bbox or len(bbox) != 4:
        return None

    try:
        x_min, y_min, w, h = map(float, bbox)
        img_w = float(img_w)
        img_h = float(img_h)
    except (TypeError, ValueError):
        return None

    if img_w <= 0 or img_h <= 0 or w <= 0 or h <= 0:
        return None

    x_center = (x_min + w / 2.0) / img_w
    y_center = (y_min + h / 2.0) / img_h
    norm_w = w / img_w
    norm_h = h / img_h

    x_center = min(max(x_center, 0.0), 1.0)
    y_center = min(max(y_center, 0.0), 1.0)
    norm_w = min(max(norm_w, 0.0), 1.0)
    norm_h = min(max(norm_h, 0.0), 1.0)

    return x_center, y_center, norm_w, norm_h


def resolve_image_path(image_name, images_root: Path):
    raw_name = str(image_name).strip()
    normalized = raw_name.replace("\\", "/")
    basename = Path(normalized).name

    search_paths = [
        images_root / raw_name,
        images_root / normalized,
        images_root / basename,
    ]

    for path in search_paths:
        if path.exists():
            return path

    matches = list(images_root.rglob(basename))
    if matches:
        return matches[0]

    return None


def main():
    print(f"Loading JSON from: {RAW_JSON_PATH}")
    print(f"Reading images from: {RAW_IMAGES_DIR}")

    data = load_json(RAW_JSON_PATH)

    if not RAW_IMAGES_DIR.exists():
        print(f"ERROR: Image folder not found: {RAW_IMAGES_DIR}")
        return

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    if not images:
        print("ERROR: No images found in JSON.")
        return

    if not annotations:
        print("ERROR: No annotations found in JSON.")
        return

    category3_lookup = find_category3_lookup(data)
    disease_map = build_disease_map_from_category3_lookup(category3_lookup)

    if not disease_map:
        print("ERROR: No disease map found from categories_3.")
        return

    print("\nDetected category_id_3 disease map:")
    for k, v in sorted(disease_map.items()):
        name = category3_lookup.get(k, "unknown")
        print(f"  category_id_3={k} -> YOLO class {v} ({name})")

    category3_counter = Counter()
    for ann in annotations:
        if "category_id_3" in ann:
            category3_counter[ann["category_id_3"]] += 1

    print("\ncategory_id_3 value counts:")
    for cat_id, count in sorted(category3_counter.items()):
        name = category3_lookup.get(cat_id, "unknown")
        print(f"  {cat_id}: {count} annotations, name={name}")

    setup_directories()

    img_lookup = {}
    for img in images:
        img_id = img.get("id")
        if img_id is not None:
            img_lookup[img_id] = img

    annotations_by_img = defaultdict(list)
    for ann in annotations:
        img_id = ann.get("image_id")
        if img_id is not None:
            annotations_by_img[img_id].append(ann)

    print("\nConverting annotations to YOLO format...")

    converted_images = 0
    total_labels = 0
    missing_images = 0
    missing_metadata = 0
    invalid_boxes = 0
    skipped_without_target_disease = 0

    for img_id, anns in annotations_by_img.items():
        img_info = img_lookup.get(img_id)
        if not img_info:
            missing_metadata += 1
            continue

        img_filename = img_info.get("file_name") or img_info.get("filename")
        img_w = img_info.get("width")
        img_h = img_info.get("height")

        if not img_filename or img_w is None or img_h is None:
            missing_metadata += 1
            continue

        src_img_path = resolve_image_path(img_filename, RAW_IMAGES_DIR)
        if src_img_path is None:
            missing_images += 1
            continue

        yolo_lines = []
        seen_lines = set()

        for ann in anns:
            disease_id = ann.get("category_id_3")
            if disease_id not in disease_map:
                continue

            yolo_class = disease_map[disease_id]
            yolo_box = coco_bbox_to_yolo(ann.get("bbox"), img_w, img_h)

            if yolo_box is None:
                invalid_boxes += 1
                continue

            x_center, y_center, norm_w, norm_h = yolo_box
            line = f"{yolo_class} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"

            if line not in seen_lines:
                seen_lines.add(line)
                yolo_lines.append(line)

        if not yolo_lines:
            skipped_without_target_disease += 1
            continue

        out_img_name = Path(src_img_path.name).name
        out_txt_name = f"{Path(out_img_name).stem}.txt"

        shutil.copy2(src_img_path, IMAGES_TRAIN / out_img_name)

        with open(LABELS_TRAIN / out_txt_name, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines))

        converted_images += 1
        total_labels += len(yolo_lines)

    print("\nDONE")
    print(f"Converted images: {converted_images}")
    print(f"Total labels written: {total_labels}")
    print(f"Missing images: {missing_images}")
    print(f"Missing image metadata: {missing_metadata}")
    print(f"Invalid boxes skipped: {invalid_boxes}")
    print(f"Images skipped with no target disease: {skipped_without_target_disease}")
    print(f"Output folder: {YOLO_BASE_DIR}")


if __name__ == "__main__":
    main()