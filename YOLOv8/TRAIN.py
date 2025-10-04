from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    model = YOLO(r"C:\YOLO\yolov8s.pt") 
    model.train(
        data=r"C:\YOLOTEST\rock-paper-scissors.v1i.yolov8\data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,    
        workers=0    
    )

if __name__ == "__main__":
    freeze_support()
    main()
