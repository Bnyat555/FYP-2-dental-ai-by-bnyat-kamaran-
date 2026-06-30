from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8s.pt")

    model.train(
        data="data/detect_teeth/dental_detect.yaml",
        imgsz=1024,
        epochs=150,
        batch=2,
        workers=0,
        patience=35,
        close_mosaic=15,
        cos_lr=True,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        pretrained=True,
        amp=False,
        cache=False,
        device="cpu",
        project="runs",
        name="tooth_detect_v2"
    )