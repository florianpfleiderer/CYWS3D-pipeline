#!/bin/sh
# created by Florian Pfleiderer

# initialise submodule
git submodule update --init --recursive

base_name="cyws-3d.ckpt"
gzipped_name="$base_name.gz"
file_url="https://thor.robots.ox.ac.uk/cyws-3d/$gzipped_name"

# Check if the base file already exists
if [ -e "$base_name" ]; then
    echo "File '$base_name' already exists. Skipping download."
else
    # Download the gzipped file if the base file doesn't exist
    echo "Downloading $gzipped_name..."
    wget -nc "$file_url"

    # Check if the download was successful
    if [ $? -eq 0 ]; then
        echo "Download successful. Unzipping and deleting the compressed file."
        gzip -d "$gzipped_name"
    else
        echo "Error: Unable to download the file."
    fi
fi

# setup python environment
pip install -e .. 

# # download pretrained model
# wget https://thor.robots.ox.ac.uk/cyws-3d/cyws-3d.ckpt.gz

# # unzip and delete
# gzip -d cyws-3d.ckpt.gz