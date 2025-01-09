#!/bin/bash
source .env  # Load environment variables

# Use full path to conda activate
source ~/anaconda3/etc/profile.d/conda.sh  # Adjust path if needed
conda activate video-retalking

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export FLASK_ENV=${FLASK_ENV:-production}

# Start the Flask server
python app.py