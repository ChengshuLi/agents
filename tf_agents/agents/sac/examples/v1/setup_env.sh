#!/bin/bash
# This script create a new environment folder, with default (1) model parameters & training setting config (2) gibson environment & agent config (3) pedestrian / layout config (different versions)

# Experiment results are also saved under the same directory.

# Motivated by MuseGAN project

#TODO put multiple experiments' results on the same tensorboard log

# Read experiment's name.
# read -p "Experiment: " exp_name
# DIR="${0%/*}/experiments/$exp_name"
while true; do
    read -p "Experiment: " exp_name
    DIR="${0%/*}/experiments/$exp_name"
    if [ ! -d "$DIR" ]; then
        mkdir -p "$DIR"
        break
    else
        read -p "Experiment $exp_name already exists; do you want to replace it? [y/n]: " choice
        case "$choice" in
            y|Y ) rm -r $DIR; echo "Replaced $DIR"; 
                mkdir -p "$DIR";
                break;;
            n|N ) ;;# read -p "Experiment: " exp_name
        esac
    fi
done


# Read layout choice.
PS3="Layout to use: "
option_layouts=("empty" "basic" "hallway")
select layout in "${option_layouts[@]}"
do
    case $layout in
        "empty")
            break
            ;;
        "basic")
            break
            ;;
        "hallway")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done

# Read movements choice.
PS3="Movements pattern: "
option_movements=("crossing" "parallel")
select movement in "${option_movements[@]}"
do
    case $movement in
        "crossing")
            break
            ;;
        "parallel")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done

# Read additional comments.
read -p "Comments: " comments
echo $1

# if [ -n "$comments" ]; then
printf "Layout: $layout\nMovements pattern: $movement\nComments: $comments" > "$DIR/comments.txt"
# echo "Movements pattern: $movement" > "$DIR/comments.txt"
# echo "Comments: $comments" > "$DIR/comments.txt"
# fi

echo $mode

cp "${0%/*}/default_configs/default_env.yaml" "$DIR/env.yaml"
cp "${0%/*}/default_configs/default_train.yaml" "$DIR/train.yaml"
cp "${0%/*}/default_configs/${layout}_layout.yaml" "$DIR/layout.yaml"
cp "${0%/*}/default_configs/${movement}.yaml" "$DIR/movements.yaml"
