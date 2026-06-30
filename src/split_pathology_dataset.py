import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(r"C:\Users\MSI\Desktop\Dental-Ai\dental_ai")
BASE = PROJECT_ROOT / "data" / "pathology"

IMAGES_TRAIN = BASE / "images" / "train"
LABELS_TRAIN = BASE / "labels" / "train"
IMAGES_VAL = BASE / "images" / "val"
LABELS_VAL = BASE / "labels" / "val"

VAL_RATIO = 0.20
SEED = 42


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main():
    if not IMAGES_TRAIN.exists() or not LABELS_TRAIN.exists():
        print("Training folders not found.")
        return

    image_files = []
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp"]

    for pattern in patterns:
        image_files.extend(IMAGES_TRAIN.glob(pattern))

    image_files = sorted(image_files)

    if not image_files:
        print("No images found in train folder.")
        return

    random.seed(SEED)
    random.shuffle(image_files)

    val_count = int(len(image_files) * VAL_RATIO)
    val_images = image_files[:val_count]

    reset_dir(IMAGES_VAL)
    reset_dir(LABELS_VAL)

    moved = 0

    for img_path in val_images:
        label_path = LABELS_TRAIN / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        shutil.move(str(img_path), str(IMAGES_VAL / img_path.name))
        shutil.move(str(label_path), str(LABELS_VAL / label_path.name))
        moved += 1

    train_count = 0
    val_count_final = 0

    for pattern in patterns:
        train_count += len(list(IMAGES_TRAIN.glob(pattern)))
        val_count_final += len(list(IMAGES_VAL.glob(pattern)))

    print(f"Validation images moved: {moved}")
    print(f"Train images left: {train_count}")
    print(f"Val images: {val_count_final}")


if __name__ == "__main__":
    main()