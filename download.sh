#!/usr/bin/env bash
wget = /usr/bin/wget
echo "Downloading Berkeley Van Gogh dataset"
wget "https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/vangogh2photo.zip"
unzip -q vangogh2photo.zip -d dataset
echo "Download completed!"