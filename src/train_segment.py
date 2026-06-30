from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8s-seg.pt")

    model.train(
        data="data/segment_teeth/dental_seg.yaml",
        imgsz=1024,
        epochs=120,
        batch=2,
        workers=0,
        patience=25,
        close_mosaic=10,
        cos_lr=True,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        pretrained=True,
        amp=False,
        cache=False,
        device="cpu",
        project="runs",
        name="tooth_segmentor_v1"
    )