from ultralytics import YOLO
import os


model = YOLO("Modellek/yolov8n.pt")

# Főkép mappa
base_folder = "Kepek"

# Eredmény mappa
result_folder = "Eredmeny"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

eredmeny_file = os.path.join(result_folder, "eredmeny.txt")
osszes_file = os.path.join(result_folder, "osszes.txt")

# 2 - auto | 3 - motor | 5 - busz | 7 - teherautó/kamiom
vehicles = [2, 3, 5, 7]

# COCO class ID -> név
class_map = {
    2: "auto",
    3: "motor",
    5: "busz",
    7: "teherauto/kamion"
}

# Global összesítő
global_counts = {name: 0 for name in class_map.values()}

# Eredmeny.txt törlése / újraírása
with open(eredmeny_file, "w", encoding="utf-8") as f:
    f.write("Képenkénti dátum szerinti eredmények:\n\n")

# Bejárjuk a Kepek/ mappát
for subfolder in sorted(os.listdir("Kepek")):

    full_path = os.path.join(base_folder, subfolder)

    # Csak mappákra dolgozunk (dátum mappák)
    if os.path.isdir(full_path):

        # Dátum mappa összesítő
        date_counts = {name: 0 for name in class_map.values()}

        # Mappán belüli képek feldolgozása
        for filename in sorted(os.listdir(full_path)):
            if filename.lower().endswith(".jpg"):
                image_path = os.path.join(full_path, filename)
                print("Feldolgozás:", image_path)

                # YOLO detektálás
                results = model(image_path, classes=vehicles, imgsz=960)

                # Egy képre szóló számlálás
                for r in results[0].boxes.cls.tolist():
                    cls_id = int(r)
                    if cls_id in class_map:
                        date_counts[class_map[cls_id]] += 1
                        global_counts[class_map[cls_id]] += 1

        # Kiírás a dátum szintű eredményhez
        with open(eredmeny_file, "a", encoding="utf-8") as f:
            f.write(f"{subfolder}:\n")
            for vehicle, cnt in date_counts.items():
                f.write(f"  {vehicle}: {cnt}\n")
            f.write("\n")

# Összesített eredmény kiírása
with open(osszes_file, "w", encoding="utf-8") as f:
    f.write("Összesített járműdetektálási eredmények:\n\n")
    for vehicle, cnt in global_counts.items():
        f.write(f"{vehicle}: {cnt}\n")

print("Kész! A részletes eredmény itt található:", eredmeny_file)
print("Az összesített eredmény itt található:", osszes_file)
