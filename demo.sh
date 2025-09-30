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
JUST_3DGS="${JUST_3DGS:-False}"


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
        --just-3dgs|-j)
            JUST_3DGS="True"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 --image-path IMAGE_PATH [--offload-gpu] [--num-gpus N] [--reinstall-apex]"
            echo ""
            echo "Optional Arguments:"
            echo "  --offload-gpu, -o        Offload GPU (default: False)"
            echo "  --image-path, -i         Image path (required)"
            echo "  --num-gpus, -n           Number of GPUs (default: 1)"
            echo "  --reinstall-apex, -r     Reinstall Apex (default: False)"
            echo "  --just-3dgs, -j          Just run 3DGS (default: False)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help or -h for usage."
            exit 1
            ;;
    esac
done

if [ -z "$IMAGE_PATH" ]; then
    echo "Image path is required"
    exit 1
fi

echo "REINSTALL_APEX: $REINSTALL_APEX"
echo "OFFLOAD_GPU: $OFFLOAD_GPU"
echo "IMAGE_PATH: $IMAGE_PATH"
echo "NUM_GPUS: $NUM_GPUS"
echo "JUST_3DGS: $JUST_3DGS"

if [ "$REINSTALL_APEX" = "True" ]; then
    uv sync
    ./install_apex.sh # runs everytime because uv sync removes it
fi

if [ "$JUST_3DGS" = "False" ]; then
    echo "================ Running Latents =================="
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
fi

echo "================ Running with 3DGS =================="
command_3dgs="python sample.py --config configs/demo/lyra_static.yaml"
echo "Running command: $command_3dgs"
$command_3dgs

echo "================ Done =================="