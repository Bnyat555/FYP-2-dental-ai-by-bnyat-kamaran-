from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"runs\detect\runs\tooth_detect_v23\weights\last.pt")
    model.train(resume=True)
