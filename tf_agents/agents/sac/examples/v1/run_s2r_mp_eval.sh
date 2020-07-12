#!/bin/bash

gpu_c="1"
gpu_g="0"
algo="sac"
robot="fetch"
config_file="../examples/configs/"$robot"_interactive_nav_s2r_mp.yaml"
col="0.0"
run="0"
lr="3e-4"

log_dir="/result/test_s2r_Samuels_drawers"
echo $log_dir

python -u train_eval.py \
    --root_dir $log_dir \
    --env_type ig_s2r_mp_push_drawers \
    --config_file $config_file \
    --initial_collect_steps 20 \
    --collect_steps_per_iteration 1 \
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
