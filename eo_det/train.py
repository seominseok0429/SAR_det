from ultralytics import YOLO

model = YOLO("yolo11n-obb.yaml").load("fine_tune.pt")  

#results = model.train(data="dota8.yaml", epochs=100, imgsz=640)
