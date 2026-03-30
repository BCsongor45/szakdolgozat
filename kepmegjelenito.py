from ultralytics import YOLO
import cv2

model = YOLO("Modellek/yolov8l.pt")

photo_path = "Kepek/2025-03-26_20-21/1674124862.jpg"

vehicles = [2, 3, 5, 7]

detected_list = model(photo_path,
                      classes=vehicles,
                      imgsz=960,
                      conf=0.25,
                      iou=0.7)

annotated_photo = detected_list[0].plot()

print(annotated_photo.shape)

cv2.imshow("Detektalt kep bounding box-xal", annotated_photo)
cv2.waitKey(0)
cv2.destroyAllWindows()