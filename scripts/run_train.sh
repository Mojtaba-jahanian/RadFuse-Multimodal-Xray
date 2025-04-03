#!/bin/bash

# Activate environment if needed
# source activate radfuse-env

# Train the model using default config
python training/train.py --config config/default.yaml
