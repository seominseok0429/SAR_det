from ultralytics import YOLO

model = YOLO("fine_tune.pt")
results = model("./P0182.png")
for r in results:
    r.save(filename="P0182_vis.png")

