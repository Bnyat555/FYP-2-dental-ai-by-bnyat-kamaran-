from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8s-cls.pt")

    model.train(
        data="data/tooth_identity_32",
        imgsz=224,
        epochs=60,
        batch=16,
        workers=0,
        patience=15,
        optimizer="AdamW",
        lr0=0.001,
        pretrained=True,
        amp=False,
        cache=False,
        device="cpu",
        project="runs",
        name="tooth_identity_32_classifier"
    )