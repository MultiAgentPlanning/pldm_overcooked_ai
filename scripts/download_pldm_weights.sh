#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
id=1-gX1vJ-DkO-f1EtSTjA2xgviQ9AbjS0w
out=data/pldm2/pldm.zip
gdown --id "$id" --output "$out"
unzip -o "$out" -d data   # extract into data/
rm "$out"                 # remove zip after unpacking

# https://drive.google.com/file/d/1-gX1vJ-DkO-f1EtSTjA2xgviQ9AbjS0w/view?usp=sharing