#!/bin/bash

algo="sac"
robot="fetch"
config_file="../examples/configs/"$robot"_interactive_nav_s2r_mp.yaml"
# config_file="../examples/configs/"$robot"_interactive_nav_s2r_mp_continuous.yaml"
lr="3e-4"
gamma="0.99"
env_type="ig_s2r"
# env_type="ig_s2r_baseline"

train_checkpoint_interval="1000"
policy_checkpoint_interval="1000"
rb_checkpoint_interval="5000"
log_interval="25"
summary_interval="25"

gpu_c="1"
gpu_g="0"
model_ids="candcenter"
model_ids_eval="candcenter"
col="0.0"
arena="push_door"
seed="0"
num_parallel="1"
log_dir="test"
num_eval_episodes="100"
env_mode="headless"
fine_motion_plan="true"
base_mp_algo="birrt"  # birrt | lazy_prm
arm_mp_algo="birrt"  # birrt | lazy_prm
optimize_iter="0"

### change default arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu_c) gpu_c="$2"; shift ;;
        --gpu_g) gpu_g="$2"; shift ;;
        --model_ids) model_ids="$2"; shift ;;
        --model_ids_eval) model_ids_eval="$2"; shift ;;
        --col) col="$2"; shift ;;
        --arena) arena="$2"; shift ;;
        --seed) seed="$2"; shift ;;
        --log_dir) log_dir="$2"; shift ;;
        --num_parallel) num_parallel="$2"; shift ;;
        --num_eval_episodes) num_eval_episodes="$2"; shift ;;
        --env_mode) env_mode="$2"; shift ;;
        --fine_motion_plan) fine_motion_plan="$2"; shift ;;
        --base_mp_algo) base_mp_algo="$2"; shift ;;
        --arm_mp_algo) arm_mp_algo="$2"; shift ;;
        --optimize_iter) optimize_iter="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "log_dir:" $log_dir
echo "model_ids_eval:" $model_ids_eval
echo "arena:" $arena
echo "seed:" $seed
echo "num_eval_episodes:" $num_eval_episodes
echo "env_mode:" $env_mode
echo "fine_motion_plan:" $fine_motion_plan
echo "base_mp_algo:" $base_mp_algo
echo "arm_mp_algo:" $arm_mp_algo
echo "optimize_iter:" $optimize_iter

python -u train_eval.py \
    --root_dir $log_dir \
    --env_type $env_type \
    --arena $arena \
    --config_file $config_file \
    --initial_collect_steps 200 \
    --collect_steps_per_iteration 1 \
    --num_iterations 100000000 \
    --batch_size 256 \
    --train_steps_per_iteration 1 \
    --replay_buffer_capacity 10000 \
    --train_checkpoint_interval $train_checkpoint_interval \
    --policy_checkpoint_interval $policy_checkpoint_interval \
    --rb_checkpoint_interval $rb_checkpoint_interval \
    --log_interval $log_interval \
    --summary_interval $summary_interval \
    --num_eval_episodes $num_eval_episodes \
    --eval_interval 100000000 \
    --gpu_c $gpu_c \
    --gpu_g $gpu_g \
    --num_parallel_environments 1 \
    --num_parallel_environments_eval $num_parallel \
    --actor_learning_rate $lr \
    --critic_learning_rate $lr \
    --alpha_learning_rate $lr \
    --gamma $gamma \
    --model_ids $model_ids \
    --model_ids_eval $model_ids_eval \
    --collision_reward_weight $col \
    --fine_motion_plan=$fine_motion_plan \
    --base_mp_algo $base_mp_algo \
    --arm_mp_algo $arm_mp_algo \
    --optimize_iter $optimize_iter \
    --env_mode $env_mode \
    --eval_only
    # --eval_deterministic \
    # > $log_dir/log 2>&1