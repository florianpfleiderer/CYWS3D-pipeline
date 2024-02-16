import os
from matplotlib.pylab import f
import yaml
from itertools import groupby, zip_longest

# Pfad zum Verzeichnis mit den Bildern
image_dir = "."
folder = "office/"

# Liste für die Bildinformationen
batch = []

# Sammeln und sortieren Sie die Dateien im Verzeichnis
images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
images.sort(key=lambda x: x[3:5])

# Gruppieren Sie die Bilder nach Bildnummer
grouped_images = groupby(images, key=lambda x: x[4])

# Durchlaufen Sie die Gruppen und erstellen Sie Paare von Bildern mit unterschiedlichen Szenen
for _, group in grouped_images:
    group = list(group)
    # print(group)
    for image1, image2 in zip_longest(group[::2], group[1::2]):
        if image1 and image2:
            batch.append({
                "image1": f"{folder}{image1}",
                "image2": f"{folder}{image2}",
                "registration_strategy": "2d"
            })

# Schreiben Sie die batch-Liste in die YAML-Datei
with open("input_metadata.yml", "w") as file:
    yaml.dump({"batch": batch}, file)