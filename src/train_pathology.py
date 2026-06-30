from pathlib import Path
from ultralytics import YOLO


PROJECT_ROOT = Path(r"C:\Users\MSI\Desktop\Dental-Ai\dental_ai")
DATA_YAML = PROJECT_ROOT / "data" / "pathology" / "data.yaml"


def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        imgsz=1024,
        epochs=80,
        batch=8,
        project=str(PROJECT_ROOT / "runs" / "pathology"),
        name="pathology_v1",
        patience=20,
        workers=2,
        pretrained=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()