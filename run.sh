#!/bin/bash
# Created on Mon May 13 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien

# rooms=("GH30_LivingArea" "GH30_Office" "GH30_SmallRoom") 
rooms=("Office")
perspectives=("null" "3d" "2d") # replace with your actual perspectives
depths=("true" "false") # replace with your actual depths

runs=1

for room in "${rooms[@]}"; do
  if [ ! -d "/data/GH30_${room}/predictions" ]; then
    mkdir -p "/data/GH30_${room}/predictions"
  fi
  for perspective in "${perspectives[@]}"; do
    for depth in "${depths[@]}"; do
      for run in $(seq 1 $runs); do
        rm -r "data/GH30_${room}/predictions/"*

        create_inference_metadata.py --room "$room" --perspective "$perspective" --depth "$depth" 
        inference.py --room "$room"

        # Copy the predictions to a new folder
        config_file="data/GH30_${room}/predictions/metadata_configurations.yaml"
        if [ -f "$config_file" ]; then
          # Read the room, perspective, and depth from the configuration file
          room_key=$(grep 'room:' "$config_file" | awk '{print $2}')
          perspective_key=$(grep 'perspective:' "$config_file" | awk '{print $2}')
          depth_key=$(grep 'depth:' "$config_file" | awk '{print $2}')
          run_number=$(printf "%02d" $run)
          current_date=$(date +%m-%d)
          key="GH30_${room_key}_${current_date}_${perspective_key}_${depth_key}_${run_number}"

          # copy predictions
          mkdir -p "data/results/${key}/predictions"
          cp -r "data/GH30_${room}/predictions/"* "data/results/${key}/predictions"
          cp -r "data/GH30_${room}/all_target_bboxes.pt" "data/results/${key}/"
          cp -r "data/GH30_${room}/input_metadata.yaml" "data/results/${key}/"
        fi
      done
    done
  done
done