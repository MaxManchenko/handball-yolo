#!/bin/bash

# Set paths for the folders
TEST_DIR="./data/test"
DATA_DIR="./data/scenes"

# Loop through all files in the test directory and its subdirectories
find "$TEST_DIR" -type f | while read -r file; do
    # Extract the relative path
    relative_path="${file#$TEST_DIR/}"
    # Check if the file exists in the data directory
    if [[ -f "$DATA_DIR/$relative_path" ]]; then
        # If it exists, remove it
        rm "$DATA_DIR/$relative_path"
        echo "Deleted: $DATA_DIR/$relative_path"
    fi
done
