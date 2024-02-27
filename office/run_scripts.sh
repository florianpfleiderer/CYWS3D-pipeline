#/bin/bash
# Check if the argument is provided
if [ $# -eq 0 ]; then
    # Argument not provided, use default value
    ARGUMENT=0
else
    # Argument provided, use the provided value
    ARGUMENT="$1"
fi

# run greyscale conversion
python3 convert_to_greyscale.py

# run yml creator
python3 create_yml.py "$ARGUMENT"

