#!/bin/sh

echo "=== STEP 0: Pull repository ==="

# check if GIT_PAT is set
if [ -z "$GIT_PAT" ]; then
  echo "GIT_PAT is not set. Please set it to your GitHub personal access token."
  exit 1
fi

export ACCOUNT="Tremirre"
export REPO_NAME="event-sun-effects-remover"
export REPO_URL="https://${GIT_PAT}@github.com/${ACCOUNT}/${REPO_NAME}.git"

git clone ${REPO_URL} --depth 1
cd ${REPO_NAME}

# Install dependencies
echo "=== STEP 1: Install dependencies ==="

sudo apt-get update
sudo apt-get install -y vim ffmpeg libsm6 libxext6 unzip
pip install -r requirements.prod.txt

DIR_DATA="/workspace/"
echo "=== STEP 2: Setup data directory ==="
mkdir ./data
unzip ${DIR_DATA}/proc-all.zip -d ./data
unzip ${DIR_DATA}/ref.zip -d ./data
mkdir ./data/detect/
unzip ${DIR_DATA}/flares.zip -d ./data/detect/

echo "=== STEP 3: Add cwd to PYTHONPATH ==="
export PYTHONPATH=${PWD}

echo "=== STEP 4: Copy .env file ==="
cp ${DIR_DATA}/.env .

echo "=== STEP 5: Split data files ==="
python -m src.split-data

echo "=== DONE! ==="

