from ultralytics import YOLO
import os
import csv


model = YOLO("Modellek/yolov8n.pt")


base_folder = "Kepek"

result_folder = "Eredmeny"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

datum_folder = os.path.join(result_folder, "Datum")
if not os.path.exists(datum_folder):
    os.makedirs(datum_folder)


vehicles = [2, 3, 5, 7]
class_map = {
    2: "auto",
    3: "motor",
    5: "busz",
    7: "teherauto/kamion"
}

global_counts = {name: 0 for name in class_map.values()}


for subfolder in sorted(os.listdir(base_folder)):
    full_path = os.path.join(base_folder, subfolder)

    if os.path.isdir(full_path):
        date_counts = {name: 0 for name in class_map.values()}

        for filename in sorted(os.listdir(full_path)):
            if filename.lower().endswith(".jpg"):
                image_path = os.path.join(full_path, filename)

                results = model(image_path, classes=vehicles, imgsz=960)

                for res in results[0].boxes.cls.tolist():
                    class_id = int(res)
                    if class_id in class_map:
                        name = class_map[class_id]
                        date_counts[name] += 1
                        global_counts[name] += 1

        date_csv_path = os.path.join(datum_folder, f"{subfolder}.csv")

        with open(date_csv_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["datum", subfolder])
            writer.writerow(["jarmu", "db"])
            for vehicle_type, count in date_counts.items():
                writer.writerow([vehicle_type, count])

osszes_csv = os.path.join(result_folder, "osszes.csv")

with open(osszes_csv, "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["jarmu", "db"])
    for vehicle, count in global_counts.items():
        writer.writerow([vehicle, count])

print("Kész!")
