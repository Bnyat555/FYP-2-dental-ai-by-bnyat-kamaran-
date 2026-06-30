from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8n-cls.pt")

    model.train(
        data="data/quadrant_cls",
        imgsz=224,
        epochs=50,
        batch=32,
        workers=0,
        project="runs",
        name="quadrant_classifier"
    )
