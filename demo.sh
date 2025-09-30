#!/bin/bash

set -e

# Make sure that install_apex.sh was invoked before running this script
if [ ! -d "dependencies/apex/build" ]; then
    echo "apex is not installed, exiting. Please run install_apex.sh first"
    exit 1
fi

# Make sure that download_models.sh was invoked before running this script
if [ ! -d "checkpoints" ]; then
    echo "checkpoints are not downloaded, exiting. Please run download_models.sh first"
    exit 1
fi

OFFLOAD_GPU="${OFFLOAD_GPU:-False}"
IMAGE_PATH="${IMAGE_PATH:-}"
NUM_GPUS="${NUM_GPUS:-1}"
REINSTALL_APEX="${REINSTALL_APEX:-False}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --offload-gpu|-o)
            OFFLOAD_GPU="True"
            shift
            ;;
        --image-path|-i)
            IMAGE_PATH="$2"
            shift 2
            ;;
        --num-gpus|-n)
            NUM_GPUS="$2"
            shift 2
            ;;
        --reinstall-apex|-r)
            REINSTALL_APEX="True"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 --image-path IMAGE_PATH [--offload-gpu]"
            echo ""
            echo "Optional Arguments:"
            echo "  --offload-gpu, -o  Offload GPU (default: False)"
            echo "  --image-path, -i  Image path (required)"
            echo "  --num-gpus, -n    Number of GPUs (default: 1)"
            echo "  --reinstall-apex, -r  Reinstall Apex (default: False)"
            exit 0
            ;;
    esac
    shift
done

if [ -z "$IMAGE_PATH" ]; then
    echo "Image path is required"
    exit 1
fi

if [ "$REINSTALL_APEX" = "True" ]; then
    uv sync
    ./install_apex.sh # runs everytime because uv sync removes it
fi

command="python cosmos_predict1/diffusion/inference/gen3c_single_image_sdg.py \
    --checkpoint_dir checkpoints \
    --num_gpus $NUM_GPUS \
    --input_image_path $IMAGE_PATH \
    --video_save_folder assets/demo/static/diffusion_output_generated \
    --foreground_masking \
    --multi_trajectory"

if [ "$OFFLOAD_GPU" = "True" ]; then
    command="$command --offload_diffusion_transformer \
    --offload_tokenizer \
    --offload_text_encoder_model \
    --offload_prompt_upsampler \
    --offload_guardrail_models \
    --disable_guardrail \
    --disable_prompt_encoder"
fi

echo "Running command: $command"
$command