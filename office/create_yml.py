import os
from turtle import left
from matplotlib.pylab import f
import yaml
from itertools import groupby, zip_longest
import logging
import argparse

# Konfigurieren Sie das Logging-Modul, um Informationen auf der Konsole anzuzeigen
logging.basicConfig(level=logging.INFO)

# Pfad zum Verzeichnis mit den Bildern
image_dir = "."
folder = "office/"

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("arg1", nargs='?', default=None, help="Optional argument")
args = parser.parse_args()

# Use the value of arg1 in your script
if args.arg1:
    # Do something with arg1
    print(f"arg1={args.arg1}")

# Liste für die Bildinformationen
batch = []
depth_images = []

# Sammeln und sortieren Sie die Dateien im Verzeichnis
images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
images.sort(key=lambda x: x[3:5])

# Extrahieren Sie alle Tiefenbilder in eine separate Liste
depth_images = [image for image in images if 'depth' in image]

# Entfernen Sie die Tiefenbilder aus der images-Liste
images = list(filter(lambda x: 'depth' not in x, images))

# Gruppieren Sie die Bilder nach Bildnummer
grouped_images = groupby(images, key=lambda x: x[4])

cntr = 0 # prediction counter

# Durchlaufen Sie die Gruppen und erstellen Sie Paare von Bildern mit unterschiedlichen Szenen
for _, group in grouped_images:
    group = list(group)
    logging.info(f"########### Processing group {group} with {len(group)} images ###########")
    for i in range(len(group)):
        for j in range(i+1, len(group)):
            image1 = group[i]
            image2 = group[j]
            if args.arg1[0] == "L" and image1[9] == "L" and image2[9] == "L":
                # Überprüfen Sie, ob ein Tiefenbild mit demselben Namen existiert
                if args.arg1[1] == "0" and any(image1[:-4] in depth_image for depth_image in depth_images) and any(image2[:-4] in depth_image for depth_image in depth_images):
                    batch.append({
                        "image1": f"{folder}{image1}",
                        "image2": f"{folder}{image2}",
                        "depth1": f"{folder}{image1[:-4]}_depth.png",
                        "depth2": f"{folder}{image2[:-4]}_depth.png",
                        "registration_strategy": "3d"
                    })
                    logging.info(f"Fixed-Angle Pair {cntr}: {image1} and {image2} with 3d registration")
                    cntr += 1
                elif args.arg1[1] == "1":
                    batch.append({
                        "image1": f"{folder}{image1}",
                        "image2": f"{folder}{image2}",
                        "registration_strategy": "2d"
                    })
                    logging.info(f"Fixed-Angle Pair {cntr}: {image1} and {image2} with 2d registration")
                    cntr += 1
                else:
                    logging.warning(f"wrong args: L0 for 3d registration; L1 for 2d registration")
                
            elif args.arg1 == "R" and image1[9] == "L" and image2[9] == "R" and image1[7] != image2[7]:
                # Überprüfen Sie, ob ein Tiefenbild mit demselben Namen existiert
                if any(image1[:-4] in depth_image for depth_image in depth_images) and any(image2[:-4] in depth_image for depth_image in depth_images):
                    batch.append({
                        "image1": f"{folder}{image1}",
                        "image2": f"{folder}{image2}",
                        "depth1": f"{folder}{image1[:-4]}_depth.png",
                        "depth2": f"{folder}{image2[:-4]}_depth.png",
                        "registration_strategy": "3d"
                    })
                    logging.info(f"Varied-Angle Pair {cntr}: {image1} and {image2}")
                    cntr += 1
                else:
                    logging.warning(f"Depth image for {image1} or {image2} not found")
            else:
                # logging.info(f"Skipping {image1} and {image2}")
                pass

# Schreiben Sie die batch-Liste in die YAML-Datei
with open("input_metadata.yml", "w") as file:
    yaml.dump({"batch": batch}, file)