import os
from PIL import Image

# Durch alle Dateien im aktuellen Verzeichnis iterieren
for filename in os.listdir('.'):
    # Nur PNG-Dateien, die "depth" im Namen haben, verarbeiten
    if 'depth' in filename and filename.endswith('.png'):
        # Bild laden
        img = Image.open(filename)

        # Bild in Graustufen konvertieren
        img_gray = img.convert('L')

        # Graustufenbild speichern
        img_gray.save(filename)
        print(f"Converted {filename} to greyscale")