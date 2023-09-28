from ultralytics import YOLO
from PIL import Image


model = YOLO('yolov8m.pt')
result = model.predict("download1.jpg")

result = result[0]
print(len(result.boxes))

for box in result.boxes:
    label = result.names[box.cls[0].item()]
    cords = [round(x) for x in box.xyxy[0].tolist()]
    prob = box.conf[0].item()
    print("Object type: ", label)
    print("coordinates: ", cords)
    print("probabilty: ", prob)

img = Image.fromarray(result.plot()[:,:,::-1])
img.show()