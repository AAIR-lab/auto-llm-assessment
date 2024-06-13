#!/bin/bash

BASE_DIR="./experiments"
MAX_BATCHES=5

export PYTHONHASHSEED=0

for ((i=0; i < $MAX_BATCHES; i++)); do
    python3 dataset_generator.py --filter-field num_operators --dataset-type plogic --base-dir "$BASE_DIR"/"batch$i"/"plogic"/
    python3 dataset_generator.py --filter-field num_operators --dataset-type ksat --base-dir "$BASE_DIR"/"batch$i"/"ksat"/
    python3 dataset_generator.py --filter-field num_operators --dataset-type fol --base-dir "$BASE_DIR"/"batch$i"/"fol"/
    python3 dataset_generator.py --filter-field num_operators --dataset-type fol_human --base-dir "$BASE_DIR"/"batch$i"/"fol_human"/
    python3 dataset_generator.py --filter-field depth --dataset-type regex --base-dir "$BASE_DIR"/"batch$i"/"regex"/
done
