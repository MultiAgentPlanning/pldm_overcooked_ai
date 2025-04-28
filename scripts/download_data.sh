#!/bin/bash
set -e

mkdir -p ./data
cd ./data

# Install gdown if not present
if ! command -v gdown &> /dev/null; then
    pip install gdown
fi

# Folder ID from Google Drive folder link
FOLDER_ID="1aGV8eqWeOG5BMFdUcVoP2NHU_GFPqi57"

# Recursively download contents
gdown --folder "https://drive.google.com/drive/folders/${FOLDER_ID}"
