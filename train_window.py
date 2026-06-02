from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.train(
    data=r'D:\window_yolo_train\window.yaml',
    imgsz=640,
    epochs=80,
    batch=8,
    workers=0,
    project=r'D:\window_yolo_train\runs',
    name='window_yolov8n_640',
    exist_ok=True,
)
