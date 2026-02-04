#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robot

export OMNI_LOG_LEVEL=ERROR
export CARB_LOG_LEVEL=ERROR
export CUDA_MODULE_LOADING=LAZY

#  Vulkan on NVIDIA GPU (skip AMD iGPU)
if [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

python "$@"