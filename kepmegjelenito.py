from ultralytics import YOLO
import cv2


model = YOLO("Modellek/yolov8n.pt")

photo_path = "Kepek/2025.03.26 12-13/1674118688.jpg"
# photo_path = "Kepek/2025.03.26 20.21/1674124862.jpg" # Kép a kevésbé találathoz

vehicles = [2, 3, 5, 7]

detected_list = model(photo_path, classes=vehicles, imgsz=960)

# detected_list = model(photo_path, classes=vehicles, imgsz=960, conf=0.1) # Alapértéke 0.25

annotated_photo = detected_list[0].plot()

cv2.imshow("Detektalt kep bounding box-xal", annotated_photo)
cv2.waitKey(0)
cv2.destroyAllWindows()