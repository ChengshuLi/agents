#!/bin/bash

gpu_c="1"
gpu_g="0"
algo="sac"
robot="fetch"
config_file="../examples/configs/"$robot"_interactive_nav_s2r_mp_continuous.yaml"
col="0.0"
run="0"
lr="3e-4"

log_dir="/result/flat_baseline_push_door"
echo $log_dir

#python -u train_eval.py \
#    --root_dir $log_dir \
#    --env_type ig_s2r_mp_empty_baseline \
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
#    --env_mode "gui"
#exit

nohup python -u train_eval.py \
    --root_dir $log_dir \
    --env_type ig_s2r_mp_push_door \
    --config_file $config_file \
    --initial_collect_steps 200 \
    --collect_steps_per_iteration 30 \
    --batch_size 256 \
    --train_steps_per_iteration 1 \
    --replay_buffer_capacity 10000 \
    --num_eval_episodes 1 \
    --eval_interval 10000000 \
    --gpu_c $gpu_c \
    --gpu_g $gpu_g \
    --num_parallel_environments 1 \
    --actor_learning_rate $lr \
    --critic_learning_rate $lr \
    --alpha_learning_rate $lr \
    --collision_reward_weight $col > $log_dir".log"

#    --model_ids Avonia,Avonia,Avonia,candcenter,candcenter,candcenter,gates_jan20,gates_jan20 \
#    --model_ids Avonia,Avonia,Avonia,Avonia,gates_jan20,gates_jan20,gates_jan20,gates_jan20 \
