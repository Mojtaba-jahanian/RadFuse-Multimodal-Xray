#!/bin/bash

# Evaluate classification and retrieval tasks separately

# Classification evaluation
python evaluation/eval_classification.py --config config/default.yaml --checkpoint checkpoints/best_model.pt

# Retrieval evaluation
python evaluation/eval_retrieval.py --config config/default.yaml --checkpoint checkpoints/best_model.pt
