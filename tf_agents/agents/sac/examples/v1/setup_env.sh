#!/bin/bash
# This script create a new environment folder, with default (1) model parameters & training setting config (2) gibson environment & agent config (3) pedestrian / layout config (different versions)

# Experiment results are also saved under the same directory.

# Motivated by MuseGAN project

#TODO put multiple experiments' results on the same tensorboard log

read -p "Experiment: " exp_name
DIR="${0%/*}/experiments/$exp_name"

read -p "Comments: " comments
echo $1
if [ ! -d "$DIR" ]; then
    mkdir -p "$DIR"
fi

if [ -n "$comments" ]; then
    echo "$comments" > "$DIR/comments.txt"
fi

cp "${0%/*}/default_configs/default_env.yaml" "$DIR/env.yaml"
cp "${0%/*}/default_configs/default_train.yaml" "$DIR/train.yaml"
cp "${0%/*}/default_configs/default_layout.yaml" "$DIR/layout.yaml"
