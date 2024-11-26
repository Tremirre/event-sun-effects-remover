#!/bin/sh

# Install dependencies
echo "=== STEP 1: Install dependencies ==="

sudo apt-get update
sudo apt-get install -y vim ffmpeg libsm6 libxext6 unzip
pip install -r requirements.prod.txt

DIR_DATA="/workspace/"
echo "=== STEP 2: Setup data directory ==="
mkdir ./data
unzip ${DIR_DATA}/processed.zip -d ./data
unzip ${DIR_DATA}/ref.zip -d ./data

echo "=== STEP 3: Add cwd to PYTHONPATH ==="
export PYTHONPATH="${PYTHONPATH}:${PWD}"

echo "=== STEP 4: Copy .env file ==="
cp ${DIR_DATA}/.env .

echo "=== STEP 5: Setup complete ==="
echo "You can now run the app by running 'python ./src/train.py'"


