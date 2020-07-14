#!/bin/bash

gpu_c="1"
gpu_g="0"
algo="sac"
robot="fetch"
config_file="../examples/configs/"$robot"_interactive_nav_s2r_mp.yaml"
col="0.0"
run="0"
lr="3e-4"
arena="push_door"
seed="0"

### change default arguments

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu_c) gpu_c="$2"; shift ;;
        --gpu_g) gpu_g="$2"; shift ;;
        --arena) arena="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

log_dir="/result/test_s2r_sac_$arena-$seed"
echo $log_dir
echo $seed
echo $arena
echo $gpu_c
echo $gpu_g

python -u train_eval.py \
    --root_dir $log_dir \
    --env_type ig_s2r \
    --arena $arena \
    --config_file $config_file \
    --initial_collect_steps 200 \
    --collect_steps_per_iteration 1 \
    --batch_size 256 \
    --train_steps_per_iteration 1 \
    --replay_buffer_capacity 10000 \
    --num_eval_episodes 1 \
    --eval_interval 10000000 \
    --gpu_c $gpu_c \
    --gpu_g $gpu_g \
    --num_parallel_environments 9 \
    --actor_learning_rate $lr \
    --critic_learning_rate $lr \
    --alpha_learning_rate $lr \
    --model_ids Avonia,Avonia,Avonia,candcenter,candcenter,candcenter,gates_jan20,gates_jan20,gates_jan20 \
    --collision_reward_weight $col > $log_dir.log 2>&1
exit

#python -u train_eval.py \
#    --root_dir $log_dir \
#    --env_type ig_s2r_mp_button_door \
#    --config_file $config_file \
#    --initial_collect_steps 200 \
#    --collect_steps_per_iteration 1 \
#    --batch_size 256 \
#    --train_steps_per_iteration 1 \
#    --replay_buffer_capacity 10000 \
#    --num_eval_episodes 100 \
#    --eval_interval 10000000 \
#    --gpu_c $gpu_c \
#    --gpu_g $gpu_g \
#    --num_parallel_environments 1 \
#    --actor_learning_rate $lr \
#    --critic_learning_rate $lr \
#    --alpha_learning_rate $lr \
#    --collision_reward_weight $col \
#    --eval_only \
#    --env_mode gui
