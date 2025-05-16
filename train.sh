#!/bin/bash

# Create a unique log filename
LOGFILE="train_$(date +%Y%m%d_%H%M%S).log"

# Run the training script with nohup and global Python interpreter
nohup python -m src.train \
        --detector-model unet \
        --detector-kwargs n_blocks=3,block_depth=2,kernel_size=3 \
        --combiner-model masked_removal \
        --combiner-kwargs binarize=0,yuv_interpolation=0,soft_factor=0 \
        --inpainter-model unet \
        --inpainter-kwargs n_blocks=4,block_depth=3,kernel_size=3 \
        --run-tags vast \
        --max-epochs 100 \
        --frac-used 1 \
        --num-workers 4 \
        --save > "$LOGFILE" 2>&1 &

