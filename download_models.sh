#!/bin/bash

# HF_TOKEN is set in the environment

set -e

export HF_TOKEN=$HF_TOKEN

python -m scripts.download_tokenizer_checkpoints --checkpoint_dir checkpoints/cosmos_predict1 --tokenizer_types CV8x8x8-720p
python scripts/download_gen3c_checkpoints.py --checkpoint_dir checkpoints
python scripts/download_lyra_checkpoints.py --checkpoint_dir checkpoints