#!/bin/bash

gpu_c="1,2"
gpu_g="0"
algo="sac"
robot="turtlebot"
config_file="../examples/configs/"$robot"_interactive_nav_s2r.yaml"
col="-0.01"
run="0"
lr="3e-4"

log_dir="test_s2r_p2p_rgbd_lidar_seg_obj_yes_dist_5m_col_-0.01"
echo $log_dir

#python -u train_eval.py \
#    --root_dir $log_dir \
#    --env_type ig_s2r \
#    --config_file $config_file \
#    --initial_collect_steps 1 \
#    --collect_steps_per_iteration 1 \
#    --batch_size 32 \
#    --train_steps_per_iteration 1 \
#    --replay_buffer_capacity 64 \
#    --num_eval_episodes 1 \
#    --eval_interval 10000000 \
#    --gpu_c $gpu_c \
#    --gpu_g $gpu_g \
#    --num_parallel_environments 1 \
#    --actor_learning_rate $lr \
#    --critic_learning_rate $lr \
#    --alpha_learning_rate $lr \
#    --collision_reward_weight $col
#exit

nohup python -u train_eval.py \
    --root_dir $log_dir \
    --env_type ig_s2r \
    --config_file $config_file \
    --initial_collect_steps 500 \
    --collect_steps_per_iteration 1 \
    --batch_size 256 \
    --train_steps_per_iteration 1 \
    --replay_buffer_capacity 12500 \
    --num_eval_episodes 1 \
    --eval_interval 10000000 \
    --gpu_c $gpu_c \
    --gpu_g $gpu_g \
    --num_parallel_environments 7 \
    --actor_learning_rate $lr \
    --critic_learning_rate $lr \
    --alpha_learning_rate $lr \
    --collision_reward_weight $col > $log_dir".log"
