from ultralytics import YOLO
from PIL import Image


model = YOLO('yolov8m.pt')
result = model.predict("download.jpg")

result = result[0]
print(len(result.boxes))
coordinate_boxes=[]
for box in result.boxes:
    label = result.names[box.cls[0].item()]
    cords = [round(x) for x in box.xyxy[0].tolist()]
    prob = box.conf[0].item()
    coordinate_boxes.append(cords)
    print("Object type: ", label)
    print("coordinates: ", cords)
    print("probabilty: ", prob)
print(coordinate_boxes)
img = Image.fromarray(result.plot()[:,:,::-1])
#img.show()

import cv2
import numpy as np

# Load the original image
image = cv2.imread("download.jpg")

# Create a binary mask (initialize with zeros)
mask = np.zeros(image.shape[:2], dtype=np.uint8)

# Define the bounding boxes of the components to be removed
# You should obtain these bounding boxes from YOLOv8 detection results
component_bboxes = coordinate_boxes
# Draw filled rectangles on the mask over the detected components
for bbox in component_bboxes:
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = 255

# Create a white background image of the same size as the original image
#white_background = np.full_like(image, (255, 255, 255), dtype=np.uint8)

# Copy the original image to the white background, but mask out the detected components
#result_image = cv2.copyTo(image, white_background, mask=~mask)

result_image = image.copy()
result_image[mask == 255] = [255, 255, 255]
# Save the edited image
cv2.imwrite("edited_image.jpg", result_image)
