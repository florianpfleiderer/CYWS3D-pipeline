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
    print(f"\n###########Processing group {group}###########")
    if len(group) == 2:
        print(f"{len(group)} images in group")
        for image1, image2 in zip_longest(group[::2], group[1::2]):
            if image1 and image2:
                # Überprüfen Sie, ob ein Tiefenbild mit demselben Namen existiert
                if any(image1[:-4] in depth_image for depth_image in depth_images) and any(image2[:-4] in depth_image for depth_image in depth_images):
                    print(f"Found depth images for {image1} and {image2}")
                    batch.append({
                        "image1": f"{folder}{image1}",
                        "image2": f"{folder}{image2}",
                        "depth1": f"{folder}{image1[:-4]}_depth.png",
                        "depth2": f"{folder}{image2[:-4]}_depth.png",
                        "registration_strategy": "3d"
                    })
                else:
                    batch.append({
                        "image1": f"{folder}{image1}",
                        "image2": f"{folder}{image2}",
                        "registration_strategy": "2d"
                    })
    elif len(group) == 3:
        print(f"{len(group)} images in group")
        for image1, image2, image3 in zip_longest(group[::3], group[1::3], group[2::3]):
            if image1 and image2 and image3:
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        image1 = group[i]
                        image2 = group[j]
                        print(f"Processing {image1} and {image2}")
                        # Überprüfen Sie, ob ein Tiefenbild mit demselben Namen existiert
                        if any(image1[:-4] in depth_image for depth_image in depth_images) and any(image2[:-4] in depth_image for depth_image in depth_images):
                            print(f"Found depth images for {image1} and {image2}")
                            batch.append({
                                "image1": f"{folder}{image1}",
                                "image2": f"{folder}{image2}",
                                "depth1": f"{folder}{image1[:-4]}_depth.png",
                                "depth2": f"{folder}{image2[:-4]}_depth.png",
                                "registration_strategy": "3d"
                            })
                        else:
                            batch.append({
                                "image1": f"{folder}{image1}",
                                "image2": f"{folder}{image2}",
                                "registration_strategy": "2d"
                            })
    elif len(group) == 4:
        print(f"{len(group)} images in group")
        for image1, image2, image3, image4 in zip_longest(group[::4], group[1::4], group[2::4], group[3::4]):
            if image1 and image2 and image3 and image4:
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        image1 = group[i]
                        image2 = group[j]
                        print(f"Processing {image1} and {image2}")
                        # Überprüfen Sie, ob ein Tiefenbild mit demselben Namen existiert
                        if any(image1[:-4] in depth_image for depth_image in depth_images) and any(image2[:-4] in depth_image for depth_image in depth_images):
                            print(f"Found depth images for {image1} and {image2}")
                            batch.append({
                                "image1": f"{folder}{image1}",
                                "image2": f"{folder}{image2}",
                                "depth1": f"{folder}{image1[:-4]}_depth.png",
                                "depth2": f"{folder}{image2[:-4]}_depth.png",
                                "registration_strategy": "3d"
                            })
                        else:
                            batch.append({
                                "image1": f"{folder}{image1}",
                                "image2": f"{folder}{image2}",
                                "registration_strategy": "2d"
                            })
    elif len(group) == 5:
        print(f"{len(group)} images in group")
        for image1, image2, image3, image4, image5 in zip_longest(group[::5], group[1::5], group[2::5], group[3::5], group[4::5]):
            if image1 and image2 and image3 and image4 and image5:
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        image1 = group[i]
                        image2 = group[j]
                        print(f"Processing {image1} and {image2}")
                        # Überprüfen Sie, ob ein Tiefenbild mit demselben Namen existiert
                        if any(image1[:-4] in depth_image for depth_image in depth_images) and any(image2[:-4] in depth_image for depth_image in depth_images):
                            print(f"Found depth images for {image1} and {image2}")
                            batch.append({
                                "image1": f"{folder}{image1}",
                                "image2": f"{folder}{image2}",
                                "depth1": f"{folder}{image1[:-4]}_depth.png",
                                "depth2": f"{folder}{image2[:-4]}_depth.png",
                                "registration_strategy": "3d"
                            })
                        else:
                            batch.append({
                                "image1": f"{folder}{image1}",
                                "image2": f"{folder}{image2}",
                                "registration_strategy": "2d"
    
    elif len(group) == 6:
        print(f"{len(group)} images in group")
        for image1, image2, image3, image4, image5, image6 in zip_longest(group[::6], group[1::6], group[2::6], group[3::6], group[4::6], group[5::6]):
            if image1 and image2 and image3 and image4 and image5 and image6:
                for i in range(len(group)):
                    for j in range(i+1, len(group)):
                        image1 = group[i]
                        image2 = group[j]
                        print(f"Processing {image1} and {image2}")
                        # Überprüfen Sie, ob ein Tiefenbild mit demselben Namen existiert
                        if any(image1[:-4] in depth_image for depth_image in depth_images) and any(image2[:-4] in depth_image for depth_image in depth_images):
                            print(f"Found depth images for {image1} and {image2}")
                            batch.append({
                                "image1": f"{folder}{image1}",
                                "image2": f"{folder}{image2}",
                                "depth1": f"{folder}{image1[:-4]}_depth.png",
                                "depth2": f"{folder}{image2[:-4]}_depth.png",
                                "registration_strategy": "3d"
                            })
                        else:
                            batch.append({
                                "image1": f"{folder}{image1}",
                                "image2": f"{folder}{image2}",
                                "registration_strategy": "2d"

# Schreiben Sie die batch-Liste in die YAML-Datei
with open("input_metadata.yml", "w") as file:
    yaml.dump({"batch": batch}, file)