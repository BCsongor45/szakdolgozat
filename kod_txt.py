from ultralytics import YOLO
import os
import csv

def get_time_of_day(hour):
    if 6 <= hour < 12:
        return "reggel"
    elif 12 <= hour < 18:
        return "délután"
    elif 18 <= hour < 21:
        return "este"
    else:
        return "éjszaka"


model = YOLO("Modellek/yolov8l.pt")

base_folder = "Kepek"
result_folder = "Eredmeny"
os.makedirs(result_folder, exist_ok=True)
output_csv = os.path.join(result_folder, "adatbazis.txt")

vehicles = [2, 3, 5, 7]
class_map = {
    2: "auto",
    3: "motor",
    5: "busz",
    7: "kamion"
}


location_dict = {}
with open("Hely/helyszin.txt", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split(";")
        if len(parts) == 3:
            location_dict[parts[0]] = {"helyszin": parts[1], "irany": parts[2]}


image_id = 1
with open(output_csv, "w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file, delimiter=";")
    writer.writerow(["ID", "Helyszin", "Irany", "Datum", "Napszak", "Ora", "Auto_db",
                    "Motor_db", "Busz_db", "Kamion_db"])


    for subfolder in sorted(os.listdir(base_folder)):
        full_path = os.path.join(base_folder, subfolder)
        if not os.path.isdir(full_path):
            continue

        parts = subfolder.split("_")
        date = parts[0]


        try:
            hour = int(parts[1].split("-")[0])
        except:
            hour = 12

        time_of_day = get_time_of_day(hour)


        for filename in sorted(os.listdir(full_path)):
            if not filename.lower().endswith(".jpg"):
                continue

            base_name = os.path.splitext(filename)[0]
            camera_id = base_name.split()[0]

            data = location_dict.get(camera_id, {"helyszin": "ismeretlen", "irany": "ismeretlen"})
            location = data["helyszin"]
            direction = data["irany"]

            counts = {"auto": 0, "motor": 0, "busz": 0, "kamion": 0}
            image_path = os.path.join(full_path, filename)
            results = model(source=image_path,
                            classes=vehicles,
                            imgsz=960,
                            conf=0.25,
                            iou=0.7)


            if results and len(results[0].boxes) > 0:
                for res in results[0].boxes.cls.tolist():
                    class_id = int(res)
                    if class_id in class_map:
                        counts[class_map[class_id]] += 1


            writer.writerow([
                image_id,
                location,
                direction,
                date,
                time_of_day,
                hour,
                counts["auto"],
                counts["motor"],
                counts["busz"],
                counts["kamion"]
            ])

            image_id += 1