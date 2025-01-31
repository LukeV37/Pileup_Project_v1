#!/bin/bash
WORK_DIR=$(pwd)
if [ ! -d ./submodules/torch/bin ]; then
    cd submodules
    python3 -m venv torch
    cd torch
    source ./bin/activate
    pip install --upgrade pip
    pip install -r pip_requirements.txt
    cd $WORK_DIR
else
    source ./submodules/torch/bin/activate
fi
