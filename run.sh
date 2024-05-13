#!/bin/bash
# Created on Mon May 13 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien

# List of rooms
rooms=("room1" "room2" "room3") # replace with your actual room names

# List of perspectives
perspectives=("true" "3d") # replace with your actual perspectives

# Loop over each room
for room in "${rooms[@]}"; do
  # Loop over each perspective
  for perspective in "${perspectives[@]}"; do
    # Run the commands
    python create_inference_metadata.py --room "$room"
    python inference.py
    python create_inference_metadata.py --depth true --room "$room"
    python inference.py
    python create_inference_metadata.py --perspective "$perspective" --room "$room"
    python inference.py

    # Copy the predictions to a new folder
    config_file="GH30_${room}/predictions/configuration.yaml"
    if [ -f "$config_file" ]; then
      # Read the key from the configuration file
      key=$(grep 'key:' "$config_file" | awk '{print $2}') # replace 'key:' with your actual key

      # Create the new folder if it doesn't exist
      mkdir -p "$key"

      # Copy the predictions
      cp -r "GH30_${room}/predictions/"* "$key/"

      # Remove the predictions
      rm -r "GH30_${room}/predictions/"*
    fi
  done
done