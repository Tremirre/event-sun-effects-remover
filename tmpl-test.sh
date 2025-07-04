#!/bin/bash
#SBATCH -w hgx2
#SBATCH -p hgx
#SBATCH -n1
#SBATCH -c4
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mem 96GB

ID="466"
CONFIG="pretrained/ours/$ID/config.json"
MODEL="pretrained/ours/$ID/model.pth"
WORKDIR="data/test-res/$ID/"
VIDDIR="data/videos/"
FINAL_RECS_DIR="data/final/"

mkdir -p $WORKDIR

conda run -n masters --no-capture-output python -m src.test \
        --output-dir $WORKDIR \
        --config $CONFIG \
        --weights-path $MODEL \
        --batch-size 8

if [ $? -ne 0 ]; then
    echo "Test failed"
    exit 1
fi

for directory in $FINAL_RECS_DIR*; do
    if [ -d "$directory" ]; then
        echo "Processing directory: $directory"
        stem=$(basename "$directory")
        mkdir -p "$WORKDIR/$stem"

        conda run -n masters --no-capture-output python -m src.infer --config $CONFIG \
            --weights-path $MODEL \
            --output-dir "$WORKDIR$stem" \
            --input-dir "$directory" \
            --batch-size 4

        conda run -n masters --no-capture-output python -m src.eval \
            -t "$WORKDIR$stem" \
            -r "$VIDDIR$stem.mp4" \
            -o "$WORKDIR$stem/refscores.json"

        if [ $? -ne 0 ]; then
            echo "Inference failed for directory: $directory"
            exit 1
        fi
    else
        echo "No directories found in $FINAL_RECS_DIR"
    fi
done
