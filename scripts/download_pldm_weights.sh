#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
id=1_olA4ki8yjZhzqwHspWJUxexttNWrAdK
out=data/pldm/pldm.zip
gdown --id "$id" --output "$out"
unzip -o "$out" -d data   # extract into data/
rm "$out"                 # remove zip after unpacking
