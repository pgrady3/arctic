#!/bin/bash

export ARCTIC_USERNAME="patrick.grady@gatech.edu"
export ARCTIC_PASSWORD=Devass1\!m
export SMPLX_USERNAME="patrick.grady@gatech.edu"
export SMPLX_PASSWORD=Devass1\!m
export MANO_USERNAME="patrick.grady@gatech.edu"
export MANO_PASSWORD=Devass1\!m


set -e

echo "Downloading SMPLX"
mkdir -p downloads
python scripts_data/download_data.py --url_file ./bash/assets/urls/smplx.txt --out_folder downloads
unzip downloads/models_smplx_v1_1.zip
mv models body_models

echo "Downloading MANO"
python scripts_data/download_data.py --url_file ./bash/assets/urls/mano.txt --out_folder downloads

mkdir -p unpack
cd downloads
unzip mano_v1_2.zip
mv mano_v1_2/models ../body_models/mano
cd ..
mv body_models unpack
