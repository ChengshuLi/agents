# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import gin

from tf_agents.environments import gym_wrapper
from tf_agents.environments import wrappers
from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv
from gibson2.envs.motion_planner_env import MotionPlanningBaseArmEnv
from gibson2.envs.motion_planner_env import MotionPlanningBaseArmContinuousEnv
import gibson2


@gin.configurable
def load(config_file,
         model_id=None,
         collision_reward_weight=0.0,
         env_type='gibson',
         arena='push_door',
         log_dir=None,
         fine_motion_plan=None,
         base_mp_algo='birrt',
         arm_mp_algo='birrt',
         env_mode='headless',
         action_timestep=1.0 / 10.0,
         physics_timestep=1.0 / 40.0,
         device_idx=0,
         random_position=False,
         random_height=False,
         gym_env_wrappers=(),
         env_wrappers=(),
         optimize_iter=0,
         spec_dtype_map=None):
    config_file = os.path.join(os.path.dirname(gibson2.__file__), config_file)
    if env_type == 'gibson':
        if random_position:
            env = NavigateRandomEnv(
                config_file=config_file,
                mode=env_mode,
                action_timestep=action_timestep,
                physics_timestep=physics_timestep,
                device_idx=device_idx,
                random_height=random_height)
        else:
            env = NavigateEnv(
                config_file=config_file,
                mode=env_mode,
                action_timestep=action_timestep,
                physics_timestep=physics_timestep,
                device_idx=device_idx)
    elif env_type == 'ig_s2r':
        env = MotionPlanningBaseArmEnv(
            config_file=config_file,
            model_id=model_id,
            collision_reward_weight=collision_reward_weight,
            mode=env_mode,
            action_timestep=1 / 500.0,
            physics_timestep=1 / 500.0,
            device_idx=device_idx,
            arena=arena,
            log_dir=log_dir,
            fine_motion_plan=fine_motion_plan,
            base_mp_algo=base_mp_algo,
            arm_mp_algo=arm_mp_algo,
            optimize_iter=optimize_iter,
        )
    elif env_type == 'ig_s2r_baseline':
        env = MotionPlanningBaseArmContinuousEnv(
            config_file=config_file,
            model_id=model_id,
            collision_reward_weight=collision_reward_weight,
            mode=env_mode,
            action_timestep=1 / 10.0,
            physics_timestep=1 / 40.0,
            device_idx=device_idx,
            arena=arena,
            log_dir=log_dir,
        )
    else:
        assert False, 'unknown env_type: {}'.format(env_type)

    discount = env.discount_factor
    max_episode_steps = env.max_step

    return wrap_env(
        env,
        discount=discount,
        max_episode_steps=max_episode_steps,
        gym_env_wrappers=gym_env_wrappers,
        time_limit_wrapper=wrappers.TimeLimit,
        env_wrappers=env_wrappers,
        spec_dtype_map=spec_dtype_map,
        auto_reset=True
    )


@gin.configurable
def wrap_env(env,
             discount=1.0,
             max_episode_steps=0,
             gym_env_wrappers=(),
             time_limit_wrapper=wrappers.TimeLimit,
             env_wrappers=(),
             spec_dtype_map=None,
             auto_reset=True):
    for wrapper in gym_env_wrappers:
        env = wrapper(env)
    env = gym_wrapper.GymWrapper(
        env,
        discount=discount,
        spec_dtype_map=spec_dtype_map,
        match_obs_space_dtype=True,
        auto_reset=auto_reset,
        simplify_box_bounds=True
    )

    if max_episode_steps > 0:
        env = time_limit_wrapper(env, max_episode_steps)

    for wrapper in env_wrappers:
        env = wrapper(env)

    return env
