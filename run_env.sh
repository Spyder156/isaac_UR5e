#!/bin/bash
# Helper script to run Isaac Sim environment with correct setup

# Get the directory containing this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the robot conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate robot

# Set environment variables
export ROS_PACKAGE_PATH="${SCRIPT_DIR}"
export OMNI_KIT_ACCEPT_EULA="YES"

# Disable Steam Vulkan layers (causes Isaac Sim GUI to hang)
export DISABLE_VK_LAYER_VALVE_steam_fossilize_1=1
export DISABLE_VK_LAYER_VALVE_steam_overlay_1=1

# Force NVIDIA GPU for Vulkan (in case of hybrid graphics)
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# Force window to NVIDIA-connected display (HDMI-0 is primary on NVIDIA)
export DISPLAY=:0
export __NV_PRIME_RENDER_OFFLOAD=0
export __GLX_VENDOR_LIBRARY_NAME=nvidia

# Disable GNOME compositor VSync issues with offloaded rendering
export CLUTTER_PAINT=disable-clipped-redraws:disable-culling
export MUTTER_ALLOW_BYPASS_COMPOSITOR=1

# Run the provided command or default to test_env.py
if [ $# -eq 0 ]; then
    echo "Running test_env.py (headless mode)..."
    echo "Use: ./run_env.sh test_env.py --gui  for GUI mode"
    echo ""
    python -u "${SCRIPT_DIR}/test_env.py" 2>&1 | grep -v "DeprecationWarning"
else
    python -u "$@" 2>&1 | grep -v "DeprecationWarning"
fi
