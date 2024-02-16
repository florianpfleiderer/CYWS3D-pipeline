import os
from matplotlib.pylab import f
import yaml
from itertools import groupby, zip_longest

# Pfad zum Verzeichnis mit den Bildern
image_dir = "."
folder = "office/"

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

# Durchlaufen Sie die Gruppen und erstellen Sie Paare von Bildern mit unterschiedlichen Szenen
for _, group in grouped_images:
    group = list(group)
    for image1, image2 in zip_longest(group[::2], group[1::2]):
        if image1 and image2:
            print(f"Processing {image1} and {image2}")
            # Überprüfen Sie, ob ein Tiefenbild mit demselben Namen existiert
            if any(image1[:11] in depth_image for depth_image in depth_images) and any(image2[:11] in depth_image for depth_image in depth_images):
                print(f"Found depth images for {image1} and {image2}")
                batch.append({
                    "image1": f"{folder}{image1}",
                    "image2": f"{folder}{image2}",
                    "depth1": f"{folder}{image1[:11]}_depth.png",
                    "depth2": f"{folder}{image2[:11]}_depth.png",
                    "registration_strategy": "3d"
                })
            else:
                batch.append({
                    "image1": f"{folder}{image1}",
                    "image2": f"{folder}{image2}",
                    "registration_strategy": "3d"
                })

# Schreiben Sie die batch-Liste in die YAML-Datei
with open("input_metadata.yml", "w") as file:
    yaml.dump({"batch": batch}, file)