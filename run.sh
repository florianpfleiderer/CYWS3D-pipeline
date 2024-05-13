#!/bin/bash
# Created on Mon May 13 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien

bbox_areas=("100" "200" "300" "400")
keep_matching_bboxes=("false" "true")
minimum_confidence_threshold=("0.2" "0.25" "0.3" "0.35" "0.4" "0.45" "0.5")

rooms=("LivingArea" "Office" "SmallRoom")
perspectives=("null" "3d" "2d")
depths=("false" "true")

runs=5
for bbox_area in "${bbox_areas[@]}"; do
  annotate.py --bbox_area "$bbox_area"
  echo "\n#######################################\nBBOX AREA CHANGED TO $bbox_area\n#######################################\n"
  for keep_matching_bbox in "${keep_matching_bboxes[@]}"; do
    echo "\n#######################################\nKEEP MATCHING BBOXES CHANGED TO $keep_mathcing_bboxes\n#######################################\n"
    for minimum_confidence in "${minimum_confidence_threshold[@]}"; do
      echo "\n#######################################\nMINIMUM CONFIDENCE CHANGED TO $minimum_confidence\n#######################################\n"
      inference_key="bbox-area-${bbox_area}_matching-bbox-${keep_matching_bbox}_confidence-${minimum_confidence}"
      for room in "${rooms[@]}"; do
        if [ ! -d "/data/GH30_${room}/predictions" ]; then
          mkdir -p "/data/GH30_${room}/predictions"
        fi
        for perspective in "${perspectives[@]}"; do
          for depth in "${depths[@]}"; do
            for run in $(seq 1 $runs); do
              rm -r "data/GH30_${room}/predictions/"*
              rm -r "/data/results/${inference_key}/"*

              create_inference_metadata.py --room "$room" --perspective "$perspective" --depth "$depth" 
              inference.py --room "$room" --bbox_area "$bbox_area" --keep_matching_bbox "$keep_matching_bbox" --minimum_confidence "$minimum_confidence" 

              # Copy the predictions to a new folder
              config_file="data/GH30_${room}/predictions/metadata_configurations.yaml"
              if [ -f "$config_file" ]; then
                # Read the room, perspective, and depth from the configuration file
                room_key=$(grep 'room:' "$config_file" | awk '{print $2}')
                perspective_key=$(grep 'perspective:' "$config_file" | awk '{print $2}')
                depth_key=$(grep 'depth:' "$config_file" | awk '{print $2}')
                run_number=$(printf "%02d" $run)
                current_date=$(date +%m-%d)
                key="GH30_${room_key}_${current_date}_perspective-${perspective_key}_depth-${depth_key}_${run_number}"

                # copy predictions
                if [ ! -d "/data/results/${inference_key}/${key}/predictions" ]; then
                  mkdir -p "/data/results/${inference_key}/${key}/predictions"
                fi
                cp -r "data/GH30_${room}/predictions/"* "/data/results/${inference_key}/${key}/predictions"
                cp -r "data/GH30_${room}/all_target_bboxes.pt" "/data/results/${inference_key}/${key}/"
                cp -r "data/GH30_${room}/input_metadata.yaml" "/data/results/${inference_key}/${key}/"
              fi
            done
          done
        done
      done
    done
  done
done