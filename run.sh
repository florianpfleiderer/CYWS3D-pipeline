#!/bin/bash
# Created on Mon May 13 2024 by Florian Pfleiderer
# Copyright (c) 2024 TU Wien

bbox_areas=("200" "300" "400" "500")
keep_matching_bboxes=("false" "true")
minimum_confidence_threshold=("0.2" "0.3" "0.4")
registration_strategies=("2d" "3d")

rooms=("LivingArea" "Office" "SmallRoom")
perspectives=("null" "2d" "3d")
depths=("false" "true")

runs=3
rm -r "data/results/"*

for bbox_area in "${bbox_areas[@]}"; do
  echo -e "\n#######################################\nBBOX AREA CHANGED TO $bbox_area\n#######################################\n"
  annotate.py --bbox_area "$bbox_area"
  for keep_matching_bbox in "${keep_matching_bboxes[@]}"; do
    echo -e "\n#######################################\nKEEP MATCHING BBOXES CHANGED TO $keep_matching_bbox\n#######################################\n"
    for registration_strategy in "${registration_strategies[@]}"; do
      echo -e "\n#######################################\nREGISTRATION STRATEGY CHANGED TO $registration_strategy\n#######################################\n"
      for minimum_confidence in "${minimum_confidence_threshold[@]}"; do
        minimum_confidence_formatted=$(echo $minimum_confidence | sed 's/\.//g')
        echo -e "\n#######################################\nMINIMUM CONFIDENCE CHANGED TO $minimum_confidence_formatted\n#######################################\n"
        inference_key="area-${bbox_area}_matching-${keep_matching_bbox}_strategy-${registration_strategy}_confidence-${minimum_confidence_formatted}"
        for room in "${rooms[@]}"; do
          if [ ! -d "data/GH30_${room}/predictions" ]; then
            mkdir -p "data/GH30_${room}/predictions"
          fi
          echo -e "\n#######################################\nROOM $room\n#######################################\n"
          for perspective in "${perspectives[@]}"; do
            echo -e "\n#######################################\nPERSPECTIVE $perspective\n#######################################\n"
            for depth in "${depths[@]}"; do
              echo -e "\n#######################################\nDEPTH $depth\n#######################################\n"
              for run in $(seq 1 $runs); do
                echo -e "\n#######################################\nRUN $run\n#######################################\n"
                rm -r "data/GH30_${room}/predictions/"*

                create_inference_metadata.py --room "$room" --perspective "$perspective" --depth "$depth" --registration_strategy "$registration_strategy"
                inference.py --room "$room" --filter_predictions_with_area_under "$bbox_area" --keep_matching_bboxes_only "$keep_matching_bbox" --minimum_confidence_threshold "$minimum_confidence" 

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

                  if [ ! -d "data/results/${inference_key}/${key}/predictions" ]; then
                    mkdir -p "data/results/${inference_key}/${key}/predictions"
                  fi

                  echo "Copying predictions to data/results/${inference_key}/${key}/predictions"
                  cp -r "data/GH30_${room}/predictions/"* "data/results/${inference_key}/${key}/predictions"
                  cp -r "data/GH30_${room}/all_target_bboxes.pt" "data/results/${inference_key}/${key}/"
                  cp -r "data/GH30_${room}/input_metadata.yaml" "data/results/${inference_key}/${key}/"
                fi
              done
            done
          done
        done
      done
    done
  done
done