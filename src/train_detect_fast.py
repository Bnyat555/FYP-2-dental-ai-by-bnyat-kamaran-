from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"runs\detect\runs\tooth_detect_v23\weights\best.pt")

    model.train(
        data="data/detect_teeth/dental_detect.yaml",
        imgsz=768,
        epochs=40,
        batch=2,
        workers=0,
        patience=15,
        close_mosaic=10,
        cos_lr=True,
        optimizer="AdamW",
        lr0=0.0005,
        weight_decay=0.0005,
        pretrained=False,
        amp=False,
        cache=False,
        device="cpu",
        project="runs",
        name="tooth_detect_fast_v1"
    )
