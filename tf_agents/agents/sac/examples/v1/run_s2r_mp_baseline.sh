#!/bin/bash

algo="sac"
robot="fetch"
config_file="../examples/configs/"$robot"_interactive_nav_s2r_mp_continuous.yaml"
lr="3e-4"
gamma="0.9995"
env_type="ig_s2r_baseline"

train_checkpoint_interval="30000"
policy_checkpoint_interval="30000"
rb_checkpoint_interval="150000"
log_interval="750"
summary_interval="750"

gpu_c="1"
gpu_g="0"
model_ids="Avonia,Avonia,Avonia,candcenter,candcenter,candcenter,gates_jan20,gates_jan20,gates_jan20"
model_ids_eval="Avonia"
col="0.0"
arena="push_door"
seed="0"
num_parallel="16"

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
        --num_parallel) num_parallel="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

log_dir="/result/flat_rl_baseline_"$arena"_"$seed
mkdir -p $log_dir
echo $log_dir
echo $gpu_c
echo $gpu_g
echo $model_ids
echo $col
echo $arena
echo $seed

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
    --num_eval_episodes 1 \
    --eval_interval 100000000 \
    --gpu_c $gpu_c \
    --gpu_g $gpu_g \
    --num_parallel_environments $num_parallel \
    --actor_learning_rate $lr \
    --critic_learning_rate $lr \
    --alpha_learning_rate $lr \
    --gamma $gamma \
    --model_ids $model_ids \
    --model_ids_eval $model_ids_eval \
    --collision_reward_weight $col > $log_dir/log 2>&1
