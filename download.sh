#!/usr/bin/env bash

echo "Downloading Berkeley Van Gogh dataset"
wget "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/vangogh2photo.zip"
unzip -q vangogh2photo.zip -d dataset
echo "Download completed!"
