# coding=utf-8
# Copyright 2018 The TF-Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r'''Train and Eval SAC.

To run:

```bash
tensorboard --logdir $HOME/tmp/sac_v1/gym/HalfCheetah-v2/ --port 2223 &

python tf_agents/agents/sac/examples/v1/train_eval.py \
  --root_dir=$HOME/tmp/sac_v1/gym/HalfCheetah-v2/ \
  --alsologtostderr
```
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import app
from absl import flags
from absl import logging

from tf import TransformListener, transformations

import gin
import tensorflow as tf

from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.drivers import dynamic_step_driver
#from tf_agents.environments import suite_gibson
from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import py_metrics
from tf_agents.metrics import tf_metrics
from tf_agents.metrics import batched_py_metric
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import normal_projection_network
from tf_agents.networks.utils import mlp_layers
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.utils import episode_utils
from tf_agents.trajectories.time_step import TimeStep
from tensorflow.python.framework.tensor_spec import TensorSpec, BoundedTensorSpec
import numpy as np
from IPython import embed
import collections

import rospy
import message_filters
from std_msgs.msg import Float32, Int64, Float32MultiArray, Empty
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, Point
from sensor_msgs.msg import Image, CameraInfo, CompressedImage, LaserScan
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker
from actionlib_msgs.msg import GoalID

import rospkg
from cv_bridge import CvBridge, CvBridgeError
import cv2
import struct

flags.DEFINE_string('root_dir', os.getenv('TEST_UNDECLARED_OUTPUTS_DIR'),
                    'Root directory for writing logs/summaries/checkpoints.')
flags.DEFINE_multi_string('gin_file', None,
                          'Path to the gin config files.')
flags.DEFINE_multi_string('gin_param', None,
                          'Gin binding to pass through.')

flags.DEFINE_integer('num_iterations', 1000000,
                     'Total number train/eval iterations to perform.')
flags.DEFINE_integer('initial_collect_steps', 1000,
                     'Number of steps to collect at the beginning of training using random policy')
flags.DEFINE_integer('collect_steps_per_iteration', 1,
                     'Number of steps to collect and be added to the replay buffer after every training iteration')
flags.DEFINE_integer('num_parallel_environments', 1,
                     'Number of environments to run in parallel')
flags.DEFINE_integer('num_parallel_environments_eval', 1,
                     'Number of environments to run in parallel for eval')
flags.DEFINE_integer('replay_buffer_capacity', 1000000,
                     'Replay buffer capacity per env.')
flags.DEFINE_integer('train_steps_per_iteration', 1,
                     'Number of training steps in every training iteration')
flags.DEFINE_integer('batch_size', 256,
                     'Batch size for each training step. '
                     'For each training iteration, we first collect collect_steps_per_iteration steps to the '
                     'replay buffer. Then we sample batch_size steps from the replay buffer and train the model'
                     'for train_steps_per_iteration times.')
flags.DEFINE_float('gamma', 0.99,
                   'Discount_factor for the environment')
flags.DEFINE_float('actor_learning_rate', 3e-4,
                   'Actor learning rate')
flags.DEFINE_float('critic_learning_rate', 3e-4,
                   'Critic learning rate')
flags.DEFINE_float('alpha_learning_rate', 3e-4,
                   'Alpha learning rate')

flags.DEFINE_integer('num_eval_episodes', 10,
                     'The number of episodes to run eval on.')
flags.DEFINE_integer('eval_interval', 10000,
                     'Run eval every eval_interval train steps')
flags.DEFINE_boolean('eval_only', False,
                     'Whether to run evaluation only on trained checkpoints')
flags.DEFINE_boolean('eval_deterministic', False,
                     'Whether to run evaluation using a deterministic policy')
flags.DEFINE_integer('gpu_c', 0,
                     'GPU id for compute, e.g. Tensorflow.')

# Added for Gibson
flags.DEFINE_string('config_file', '../test/test.yaml',
                    'Config file for the experiment.')
flags.DEFINE_list('model_ids', None,
                  'A comma-separated list of model ids to overwrite config_file.'
                  'len(model_ids) == num_parallel_environments')
flags.DEFINE_list('model_ids_eval', None,
                  'A comma-separated list of model ids to overwrite config_file for eval.'
                  'len(model_ids) == num_parallel_environments_eval')
flags.DEFINE_float('collision_reward_weight', 0.0,
                   'collision reward weight')
flags.DEFINE_string('env_mode', 'headless',
                    'Mode for the simulator (gui or headless)')
flags.DEFINE_string('env_type', 'gibson',
                    'Type for the Gibson environment (gibson or ig)')
flags.DEFINE_float('action_timestep', 1.0 / 10.0,
                   'Action timestep for the simulator')
flags.DEFINE_float('physics_timestep', 1.0 / 40.0,
                   'Physics timestep for the simulator')
flags.DEFINE_integer('gpu_g', 0,
                     'GPU id for graphics, e.g. Gibson.')
flags.DEFINE_boolean('random_position', False,
                     'Whether to randomize initial and target position')

FLAGS = flags.FLAGS


def normal_projection_net(action_spec,
                          init_action_stddev=0.35,
                          init_means_output_factor=0.1):
    del init_action_stddev
    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        mean_transform=None,
        state_dependent_std=True,
        init_means_output_factor=init_means_output_factor,
        std_transform=sac_agent.std_clip_transform,
        scale_distribution=True)


class InferenceEngine(object):
    def __init__(
        self,
        root_dir,
        obs_keys=['sensor', 'rgb', 'depth', 'scan'],
        conv_1d_layer_params=None,
        conv_2d_layer_params=None,
        encoder_fc_layers=[256],
        actor_fc_layers=[256, 256],
        critic_obs_fc_layers=None,
        critic_action_fc_layers=None,
        critic_joint_fc_layers=[256, 256],
        # Params for target update
        target_update_tau=0.005,
        target_update_period=1,
        # Params for train
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        alpha_learning_rate=3e-4,
        td_errors_loss_fn=tf.compat.v1.losses.mean_squared_error,
        gamma=0.99,
        reward_scale_factor=1.0,
        gradient_clipping=None,
        # Params for eval
        eval_deterministic=False,
        # Params for summaries and logging
        debug_summaries=False,
        summarize_grads_and_vars=False
    ):
        '''A simple train and eval for SAC.'''
        root_dir = os.path.expanduser(root_dir)
        train_dir = os.path.join(root_dir, 'train')
        policy_dir = os.path.join(root_dir, 'train', 'policy')

        obs_spec = collections.OrderedDict({})
        if 'sensor' in obs_keys:
            obs_spec['sensor'] = BoundedTensorSpec(shape=(22,), dtype=tf.float32, name=None,
                                                   minimum=np.array(-3.4028235e+38, dtype=np.float32),
                                                   maximum=np.array(3.4028235e+38, dtype=np.float32))
        if 'rgb' in obs_keys:
            obs_spec['rgb'] = BoundedTensorSpec(shape=(96, 128, 3), dtype=tf.float32, name=None,
                                               minimum=np.array(-1.0, dtype=np.float32),
                                               maximum=np.array(1.0, dtype=np.float32))
        if 'depth' in obs_keys:
            obs_spec['depth'] = BoundedTensorSpec(shape=(96, 128, 1), dtype=tf.float32, name=None,
                                                  minimum=np.array(-1.0, dtype=np.float32),
                                                  maximum=np.array(1.0, dtype=np.float32))
        if 'scan' in obs_keys:
            obs_spec['scan'] = BoundedTensorSpec(shape=(220, 1), dtype=tf.float32, name=None,
                                                 minimum=np.array(-1.0, dtype=np.float32),
                                                 maximum=np.array(1.0, dtype=np.float32))
        time_step_spec = TimeStep(
            TensorSpec(shape=(), dtype=tf.int32, name='step_type'),
            TensorSpec(shape=(), dtype=tf.float32, name='reward'),
            BoundedTensorSpec(shape=(), dtype=tf.float32, name='discount',
                              minimum=np.array(0., dtype=np.float32), maximum = np.array(1., dtype=np.float32)),
            obs_spec
        )
        observation_spec = time_step_spec.observation
        action_spec = BoundedTensorSpec(shape=(8,), dtype=tf.float32, name=None,
                                        minimum=np.array(-1.0, dtype=np.float32),
                                        maximum=np.array(1.0, dtype=np.float32))

        glorot_uniform_initializer = tf.compat.v1.keras.initializers.glorot_uniform()

        preprocessing_layers = {}
        if 'rgb' in observation_spec:
            preprocessing_layers['rgb'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=conv_2d_layer_params,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'depth' in observation_spec:
            preprocessing_layers['depth'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=conv_2d_layer_params,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'scan' in observation_spec:
            preprocessing_layers['scan'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=conv_1d_layer_params,
                conv_2d_layer_params=None,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if 'sensor' in observation_spec:
            preprocessing_layers['sensor'] = tf.keras.Sequential(mlp_layers(
                conv_1d_layer_params=None,
                conv_2d_layer_params=None,
                fc_layer_params=encoder_fc_layers,
                kernel_initializer=glorot_uniform_initializer,
            ))

        if len(preprocessing_layers) <= 1:
            preprocessing_combiner = None
        else:
            preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

        actor_net = actor_distribution_network.ActorDistributionNetwork(
            observation_spec,
            action_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            fc_layer_params=actor_fc_layers,
            continuous_projection_net=normal_projection_net,
            kernel_initializer=glorot_uniform_initializer,
        )

        critic_net = critic_network.CriticNetwork(
            (observation_spec, action_spec),
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            observation_fc_layer_params=critic_obs_fc_layers,
            action_fc_layer_params=critic_action_fc_layers,
            joint_fc_layer_params=critic_joint_fc_layers,
            kernel_initializer=glorot_uniform_initializer,
        )

        global_step = tf.compat.v1.train.get_or_create_global_step()
        tf_agent = sac_agent.SacAgent(
            time_step_spec,
            action_spec,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=actor_learning_rate),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=critic_learning_rate),
            alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=alpha_learning_rate),
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            td_errors_loss_fn=td_errors_loss_fn,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=global_step)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)

        if eval_deterministic:
            eval_py_policy = py_tf_policy.PyTFPolicy(greedy_policy.GreedyPolicy(tf_agent.policy))
        else:
            eval_py_policy = py_tf_policy.PyTFPolicy(tf_agent.policy)

        def _filter_invalid_transition(trajectories, unused_arg1):
            return ~trajectories.is_boundary()[0]
        batch_size = 1
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size=1,
            max_length=1)
        dataset = replay_buffer.as_dataset(
            num_parallel_calls=1,
            sample_batch_size=1,
            num_steps=2).apply(tf.data.experimental.unbatch()).filter(
            _filter_invalid_transition).batch(batch_size).prefetch(1)
        dataset_iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
        trajectories, unused_info = dataset_iterator.get_next()
        train_op = tf_agent.train(trajectories)

        #train_checkpointer = common.Checkpointer(
        #    ckpt_dir=train_dir,
        #    agent=tf_agent,
        #    global_step=global_step)
        policy_checkpointer = common.Checkpointer(
            ckpt_dir=policy_dir,
            policy=tf_agent.policy,
            global_step=global_step
        )

        with sess.as_default():
            # Initialize graph.
            #train_checkpointer.initialize_or_restore(sess)
            policy_checkpointer.initialize_or_restore(sess)

        self.sess = sess
        self.eval_py_policy = eval_py_policy

        obs = {}
        if 'sensor' in observation_spec:
            obs['sensor'] = np.zeros((1, 22))
        if 'rgb' in observation_spec:
            obs['rgb'] = np.zeros((1, 96, 128, 3))
        if 'depth' in observaiton_spec:
            obs['depth'] = np.zeros((1, 96, 128, 1))
        if 'scan' in observation_spec:
            obs['scan'] = np.zeros((1, 220, 1))

        # activate the session
        self.inference(obs)

    def inference(self, obs):
        import time
        start = time.time()
        with self.sess.as_default():
            time_step = TimeStep(
                np.ones(1),
                np.ones(1),
                np.ones(1),
                obs,
            )
            policy_state = ()
            action_step = self.eval_py_policy.action(time_step, policy_state)
            action = action_step.action[0]
            #print('time', time.time() - start)
            return action


def main(_):
    class InferenceROSNode():
        def __init__(self):
            #self.reset_episode()

            tf.compat.v1.enable_resource_variables()
            logging.set_verbosity(logging.INFO)

            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            conv_1d_layer_params = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
            conv_2d_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
            encoder_fc_layers = [256]
            actor_fc_layers = [256]
            critic_obs_fc_layers = [256]
            critic_action_fc_layers = [256]
            critic_joint_fc_layers = [256]

            print('conv_1d_layer_params', conv_1d_layer_params)
            print('conv_2d_layer_params', conv_2d_layer_params)
            print('encoder_fc_layers', encoder_fc_layers)
            print('actor_fc_layers', actor_fc_layers)
            print('critic_obs_fc_layers', critic_obs_fc_layers)
            print('critic_action_fc_layers', critic_action_fc_layers)
            print('critic_joint_fc_layers', critic_joint_fc_layers)

            self.engine = InferenceEngine(
                root_dir=FLAGS.root_dir,
                conv_1d_layer_params=conv_1d_layer_params,
                conv_2d_layer_params=conv_2d_layer_params,
                encoder_fc_layers=encoder_fc_layers,
                actor_fc_layers=actor_fc_layers,
                critic_obs_fc_layers=critic_obs_fc_layers,
                critic_action_fc_layers=critic_action_fc_layers,
                critic_joint_fc_layers=critic_joint_fc_layers,
                actor_learning_rate=0.0,
                critic_learning_rate=0.0,
                alpha_learning_rate=0.0,
                gamma=1.0,
                eval_deterministic=FLAGS.eval_deterministic,
            )

            self.obs = {
                'depth': np.ones((1, 96, 128, 1)),
                'rgb': np.ones((1, 96, 128, 3)),
                'scan': np.ones((1, 220, 1)),
                'sensor': np.ones((1, 22))
            }
            self.pose = None
            rospy.init_node('inference-engine') #initialize ros node

            self.tf_listener_ = TransformListener()
            print('tf_listener')

            self.policy_action_publisher = rospy.Publisher('/policy_action', Float32MultiArray, queue_size=10)
            print('policy_action_publisher')
            self.subgoal_marker_publisher = rospy.Publisher('/subgoal_marker', Marker, queue_size=10)
            self.global_plan_marker_publisher = rospy.Publisher('/global_plan_marker', Marker, queue_size=10)
            self.goal_marker_publisher = rospy.Publisher('/goal_marker', Marker, queue_size=10)
            self.cancel_move_base_publisher = rospy.Publisher('/move_base/cancel', GoalID, queue_size=10)

            self.new_episode_subscriber = rospy.Subscriber('/new_episode', Empty, self.new_episode_callback, queue_size=1)

            self.bridge = CvBridge()
            depth_sub = message_filters.Subscriber('/head_camera/depth_registered/hw_registered/image_rect', Image)#, self.depth_callback, queue_size=1)
            self.depth_min_range = 0.1
            self.depth_max_range = 3.0
            rgb_sub = message_filters.Subscriber('/head_camera/rgb/image_rect_color', Image)#, self.rgb_callback, queue_size=1)
            self.cur_rgb_pub = rospy.Publisher('/cur_rgb', Image, queue_size=10)
            self.cur_depth_pub = rospy.Publisher('/cur_depth', Image, queue_size=10)
            self.cur_scan_pub = rospy.Publisher('/cur_scan', LaserScan, queue_size=10)
            self.cur_loc_pub = rospy.Publisher('cur_loc', PoseWithCovarianceStamped, queue_size=10)
            self.cur_action_pub = rospy.Publisher('/cur_action', Float32MultiArray, queue_size=10)

            print('CvBridge')

            self.fx = 533.6015729039069
            self.cx = 317.2141439571433
            self.fy = 531.661924202436
            self.cy = 222.4564622051379

            #pose_sub = message_filters.Subscriber('/amcl_pose', PoseWithCovarianceStamped)#, self.amcl_callback, queue_size=1)
            rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, self.amcl_callback, queue_size=1)
            print('amcl_pose')

            rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.global_plan_callback, queue_size=1)
            print('global_plan')

            scan_sub = message_filters.Subscriber('/base_scan', LaserScan)#, self.scan_callback, queue_size=1)
            print('base_scan')

            ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, depth_sub, scan_sub], queue_size=1, slop=0.5)#, allow_headerless=True)
            ts.registerCallback(self.sensor_callback)
            rospy.Subscriber('/finished_action', Empty, self.finished_action_callback, queue_size=1)

            self.reset_episode()

        def fromTranslationRotation(self, translation, rotation):
            """
            :param translation: translation expressed as a tuple (x,y,z)
            :param rotation: rotation quaternion expressed as a tuple (x,y,z,w)
            :return: a :class:`numpy.matrix` 4x4 representation of the transform
            :raises: any of the exceptions that :meth:`~tf.Transformer.lookupTransform` can raise

            Converts a transformation from :class:`tf.Transformer` into a representation as a 4x4 matrix.
            """
            return np.dot(transformations.translation_matrix(translation), transformations.quaternion_matrix(rotation))

        def reset_episode(self):
            print('reset episode')
            self.sensor_ready = False
            self.global_plan_ready = False
            self.global_plan_in_map_frame = []
            self.amcl_ready = False
            self.finished_prev_action = True

            # cancel nav to previous subgoal
            cancel_move_base_msg = GoalID()
            self.cancel_move_base_publisher.publish(cancel_move_base_msg)
            rospy.sleep(rospy.Duration(2.0))
            print('reset episode done!!')

        def new_episode_callback(self, msg):
            self.reset_episode()

        def finished_action_callback(self,msg):
            print('received finished action')
            self.finished_prev_action = True

        def sensor_callback(self, rgb_msg, depth_msg, scan_msg):
            self.rgb_callback(rgb_msg)
            self.depth_callback(depth_msg)
            self.scan_callback(scan_msg)
            self.process_amcl()
            self.sensor_ready = True
            #print('set sensor ready')
            return

        def global_plan_callback(self, msg):
            # only set global plan once for the final goal
            if self.global_plan_ready:
                return
            print('set global plan')
            # cancel nav to final goal
            cancel_move_base_msg = GoalID()
            self.cancel_move_base_publisher.publish(cancel_move_base_msg)
            #print('global plan callback!')
            #t = self.tf_listener_.getLatestCommonTime("/map", "/odom")
            #position, quaternion = self.tf_listener_.lookupTransform("/map", "/odom", t)
            #mat44 = self.fromTranslationRotation(position, quaternion)
            goal = msg.poses[-1]
            poses = msg.poses[::10]

            #if len(poses) >= 10:
            #    poses = poses[:10]
            #else:
            #    while len(poses) < 10:
            #        poses.append(goal)

            #plan = []

            #for pose in poses:
                #x,y = pose.pose.position.x, pose.pose.position.y
                #print(x,y)
            #    p = pose.pose.position

                #plan_in_base = self.tf_listener_.transformPose("/base_link", pose)
            #    xy = tuple(np.dot(mat44, np.array([p.x, p.y, p.z, 1.0])))[:2]
            #    plan.append(xy)

            #goal_p = goal.pose.position
            #goal_xy = tuple(np.dot(mat44, np.array([p.x, p.y, p.z, 1.0])))[:2]

            #print(plan)
            #print(goal_xy)

            #for i in range(10):
            #    self.obs["sensor"][0,i * 2] = plan[i][0]
            #    self.obs["sensor"][0,i * 2 + 1] = plan[i][1]
            #
            #self.obs["sensor"][0,20] = goal_xy[0]
            #self.obs["sensor"][0,21] = goal_xy[1]
            #print('-' * 50)
            for pose in poses:
                p = pose.pose.position
                self.global_plan_in_map_frame.append(np.array([p.x, p.y, p.z]))

            #print('-' * 20)
            #for pt in self.global_plan_in_map_frame:
            #    print('map:', pt)
            #self.last_time_global_plan_update = time.time()
            self.global_plan_ready = True

        def amcl_callback(self, msg):
            self.cur_loc_msg = msg
            self.pose = msg.pose.pose

        def process_amcl(self):
            if not self.global_plan_ready:
                return
            #t = self.tf_listener_.getLatestCommonTime("/map", "/odom")
            #position, quaternion = self.tf_listener_.lookupTransform("/map", "/odom", t)
            #mat44 = self.fromTranslationRotation(position, quaternion)
            pos = self.pose.position
            #current_pose_in_map_frame = np.dot(mat44, np.array([pos.x, pos.y, pos.z, 1.0]))[:3]
            current_pose_in_map_frame = np.array([pos.x, pos.y, pos.z])

            dists = [np.linalg.norm(current_pose_in_map_frame - pt) for pt in self.global_plan_in_map_frame]
            #print('-' * 20, 'process_acml')
            #print('current_pose_in_map_frame', current_pose_in_map_frame)
            #for pt in self.global_plan_in_map_frame:
            #    print('gp', pt)
            closest_pt_idx = np.argmin(dists)
            #print('closest_pt_idx', closest_pt_idx)
            #print('-' * 20 + 'ENDENDEND!!!!!!!!!!!')
            shortest_path = self.global_plan_in_map_frame[closest_pt_idx:(closest_pt_idx + 10)]
            while len(shortest_path) < 10:
                shortest_path.append(shortest_path[-1])

            t = self.tf_listener_.getLatestCommonTime("/base_link", "/map")
            position, quaternion = self.tf_listener_.lookupTransform("/base_link", "/map", t)
            mat44 = self.fromTranslationRotation(position, quaternion)
            shortest_path_in_base_link_frame = [np.dot(mat44, np.append(pt, 1.0))[:3] for pt in shortest_path]

            goal = self.global_plan_in_map_frame[-1]
            goal_in_base_link_frame = np.dot(mat44, np.append(goal, 1.0))[:3]

            for i in range(10):
                self.obs['sensor'][0, i * 2] = shortest_path_in_base_link_frame[i][0]
                self.obs['sensor'][0, i * 2 + 1] = shortest_path_in_base_link_frame[i][1]

            self.obs['sensor'][0, 20:22] = goal_in_base_link_frame[:2]
            self.amcl_ready = True
            #print('set amcl ready')
            return

        """
        def amcl_callback(self, msg):
            #print('amcl_callback')
            self.pose = msg.pose.pose
            #print(time.time() - self.last_time_pose_update)
            #print(self.pose)
            self.last_time_pose_update = time.time()

            #t = self.tf_listener_.getLatestCommonTime("/base_link", "/map")
            #(trans, rot) = listener.lookupTransform('/map', 'base_link', t)
            rot = transformations.quaternion_matrix([self.pose.orientation.x, self.pose.orientation.y, self.pose.orientation.z, self.pose.orientation.w])

            #(lin, ang) = self.tf_listener_.lookupTwistFull('/base_link', '/map', '/base_link', (0,0,0), '/map', rospy.Time(0.0), rospy.Duration(0.5))
            #yaw = -rot[2]
            #lin_in_baselink = np.zeros(2)
            #lin_in_baselink[0] = np.cos(yaw) * lin[0] - np.sin(yaw) * lin[1]
            #in_in_baselink[1] = np.sin(yaw) * lin[0] + np.cos(yaw) * lin[1]

            lin_in_baselink = rot.T.dot(np.array([lin[0], lin[1], 0, 1]))[:2]

            self.obs["sensor"][0,22] = lin_in_baselink[0]
            self.obs["sensor"][0,23] = lin_in_baselink[1]
            self.obs["sensor"][0,24] = 0#ang[0]
            self.obs["sensor"][0,25] = 0#ang[1]
        """

        def rgb_callback(self, msg):
            #print('rgb_callback')
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8').astype(np.float32)
                cv_image /= 255.0
                cv_image = cv2.resize(cv_image, (128, 96))
                cv_image = cv_image[None, :, :, :]
                self.cur_rgb_msg = msg
                #self.obs['rgb'] = cv_image
                #self.last_time_rgb_update = time.time()
            except CvBridgeError as e:
                print(e)

        def depth_callback(self, msg):
            #print('depth_callback')
            try:
                # convert millimeter to meter
                cv_image = self.bridge.imgmsg_to_cv2(msg).astype(np.float32) / 1000.0
                max_range = self.depth_max_range
                min_range = self.depth_min_range
                # crop 90% middle
                cv_image = cv_image[24:(480-24), 32:(640-32)]
                cv_image = np.nan_to_num(cv_image, max_range)
                cv_image[cv_image == 0.0] = max_range
                #cv_image = np.nan_to_num(cv_image, 1.0)
                cv_image[cv_image > max_range] = max_range
                cv_image[cv_image < min_range] = min_range
                self.depth_original = np.copy(cv_image)
                #print(np.min(self.depth_original), np.max(self.depth_original), np.mean(self.depth_original), np.std(self.depth_original))
                cv_image = (cv_image - min_range) / (max_range - min_range)
                # TODO: not set depth map to max range
                #cv_image[:, :] = 1.0
                cv_image = cv2.resize(cv_image, (128, 96))
                cv_image = cv_image[None, :, :, None]
                #print(cv_image.shape, np.min(cv_image), np.max(cv_image))
                self.obs['depth'] = cv_image
                self.cur_depth_msg = msg
                #self.last_time_depth_update = time.time()
            except CvBridgeError as e:
                print(e)

            """
            depth_header_size = 12
            raw_data = msg.data[depth_header_size:]
            raw_header = msg.data[:depth_header_size]

            [compfmt, depthQuantA, depthQuantB] = struct.unpack('iff', raw_header)
            # print(compfmt, depthQuantA, depthQuantB)
            np_arr = np.fromstring(raw_data, np.uint8)
            #print(np_arr)
            #print(len(np_arr))
            image_np = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)

            depth_img_scaled = depthQuantA / (image_np.astype(np.float32)-depthQuantB)
            depth_img_scaled[image_np == 0] = 0
            depth_img_mm = (depth_img_scaled * 1000).astype(np.uint16)

            depth_img_mm = cv2.resize(depth_img_mm[30:-30, 40:-40], (80, 60))
            depth_img_m = depth_img_mm.astype(np.float32) / 1000.0
            depth_img_m[depth_img_m > 4] = 4
            depth_img_m[depth_img_m < 0.6] = -1

            self.obs['depth'] = depth_img_m[None, :, :, None]
            # print(time.time() - self.last_time_depth_update)
            self.last_time_depth_update = time.time()
            # print(np.max(depth_img_m), np.min(depth_img_m), np.mean(depth_img_m), depth_img_m.shape)
            cv2.imshow('test', (depth_img_m + 1) / 5.0)
            cv2.waitKey(10)
            """

        def scan_callback(self, msg):
            self.cur_scan_msg = msg
            #print('scan_callback')
            # TODO: might need to flip LIDAR
            ranges = np.array(msg.ranges)[::3][1:][::-1]
            max_range = 25.0
            ranges = np.nan_to_num(ranges, max_range)
            ranges[ranges > max_range] = max_range
            ranges /= max_range
            # TODO: not set scan to max range
            #ranges[:] = 1.0
            ranges = ranges[None, :, None]
            #print(ranges.shape, np.min(ranges), np.max(ranges))
            self.obs['scan'] = ranges
            #self.last_time_scan_update = time.time()

        def should_publish_action(self):
            """
            return self.last_time_depth_update is not None and \
                self.last_time_rgb_update is not None and \
                self.last_time_scan_update is not None and \
                self.last_time_global_plan_update is not None and \
                self.finished_action
            """
            return self.global_plan_ready and self.sensor_ready and self.amcl_ready and self.finished_prev_action

        def run(self):
            while not rospy.is_shutdown():
                if not self.should_publish_action():
                    time.sleep(0.2)
                    continue
                self.cur_rgb_pub.publish(self.cur_rgb_msg)
                self.cur_depth_pub.publish(self.cur_depth_msg)
                self.cur_scan_pub.publish(self.cur_scan_msg)
                self.cur_loc_pub.publish(self.cur_loc_msg)
                #print(self.obs["sensor"])
                #print('depth', np.mean(self.obs["depth"]))
                # self.obs['sensor'] = np.zeros((1, 26))
                #for i in range(10):
                #    self.obs['sensor'][0, i * 2] = i * 0.2
                #    self.obs['sensor'][0, i * 2+1] = 0.0
                # self.obs['sensor'][0, 20] = 5.0
                # self.obs['sensor'][0, 21] = 0.0
                # self.obs['sensor'][0, 22] = 0.5
                # self.obs['sensor'][0, 23] = 0.0
                # print(self.obs['sensor'][0, 0:24])

                #self.obs['depth'] = np.ones((1, 60, 80, 1)) * 3.0
                #print('depth', np.min(self.obs['depth']), np.max(self.obs['depth']), np.mean(self.obs['depth']))
                action = self.engine.inference(self.obs)
                cur_action_msg = Float32MultiArray()
                cur_action_msg.data = action
                self.cur_action_pub.publish(cur_action_msg)

                arm_img_v, arm_img_u = action[4], action[5]
                arm_img_u = np.clip(int(arm_img_u * 320 + 320.0), 32, 639 - 32) - 32
                arm_img_v = np.clip(int(arm_img_v * 240.0 + 240.0), 24, 479 - 24) - 24
                #print('uv', arm_img_u, arm_img_v)
                #arm_z = self.obs['depth'][0, arm_img_v, arm_img_u, 0]
                #arm_z = self.obs['depth'][0, arm_img_v, arm_img_u, 0] * self.depth_max_range
                print('arm_img_u', arm_img_u)
                print('arm_img_v', arm_img_v)
                arm_z = self.depth_original[arm_img_v, arm_img_u]
                print('arm_z_in_camera_frame', arm_z)
                arm_x = arm_z * (arm_img_u - self.cx) / self.fx
                arm_y = arm_z * (arm_img_v - self.cy) / self.fy

                #t = self.tf_listener_.getLatestCommonTime("/head_camera_rgb_optical_frame", "/base_link")
                #t = rospy.Time()
                self.tf_listener_.waitForTransform("/base_link", "/head_camera_rgb_optical_frame", rospy.Time(), rospy.Duration(4.0))
                position, quaternion = None, None
                while position is None:
                    try:
                        now = rospy.Time.now()
                        self.tf_listener_.waitForTransform("/base_link", "/head_camera_rgb_optical_frame", now, rospy.Duration(4.0))
                        position, quaternion = self.tf_listener_.lookupTransform("/base_link", "/head_camera_rgb_optical_frame", now)
                    except:
                        print("exception")
                        time.sleep(0.5)

                mat44 = self.fromTranslationRotation(position, quaternion)
                arm_xyz_camera = np.array([arm_x, arm_y, arm_z, 1.0])
                #print(arm_xyz_camera)
                arm_xyz_base_link = np.dot(mat44, arm_xyz_camera)[:3]
                #print(arm_xyz_base_link)

                base_subgoal_theta = (action[1] * 110.0) / 180.0 * np.pi
                base_subgoal_dist = action[2] + 1.0
                base_subgoal = np.array([np.cos(base_subgoal_theta), np.sin(base_subgoal_theta)]) * base_subgoal_dist
                base_subgoal_orn = action[3] * np.pi

                real_action = np.zeros(9)
                real_action[0] = action[0]
                real_action[1:3] = base_subgoal
                real_action[3] = base_subgoal_orn
                real_action[4:7] = arm_xyz_base_link
                real_action[7:9] = action[6:8] * 0.25

                policy_action_msg = Float32MultiArray()
                policy_action_msg.data = real_action

                use_base = real_action[0] > 0.0
                if use_base:
                    base_subgoal_marker = Marker()
                    base_subgoal_marker.header.frame_id = '/base_link'
                    base_subgoal_marker.type = base_subgoal_marker.ARROW
                    base_subgoal_marker.action = base_subgoal_marker.ADD

                    base_subgoal_marker.pose.position.x = real_action[1]
                    base_subgoal_marker.pose.position.y = real_action[2]
                    base_subgoal_marker.pose.position.z = 0.0
                    q = transformations.quaternion_from_euler(0.0, 0.0, base_subgoal_orn)
                    base_subgoal_marker.pose.orientation.x = q[0]
                    base_subgoal_marker.pose.orientation.y = q[1]
                    base_subgoal_marker.pose.orientation.z = q[2]
                    base_subgoal_marker.pose.orientation.w = q[3]
                    t = rospy.Duration()
                    base_subgoal_marker.lifetime = t
                    base_subgoal_marker.scale.x = 0.5
                    base_subgoal_marker.scale.y = 0.2
                    base_subgoal_marker.scale.z = 0.0
                    base_subgoal_marker.color.a = 1.0
                    base_subgoal_marker.color.r = 1.0
                    self.subgoal_marker_publisher.publish(base_subgoal_marker)
                else:
                    arm_subgoal_marker = Marker()
                    arm_subgoal_marker.header.frame_id = '/base_link'
                    arm_subgoal_marker.type = arm_subgoal_marker.ARROW
                    arm_subgoal_marker.action = arm_subgoal_marker.ADD

                    arm_subgoal_marker.pose.position.x = real_action[4]
                    arm_subgoal_marker.pose.position.y = real_action[5]
                    arm_subgoal_marker.pose.position.z = real_action[6]
                    q = transformations.quaternion_from_euler(0.0, 0.0, np.arctan2(real_action[8], real_action[7]))
                    arm_subgoal_marker.pose.orientation.x = q[0]
                    arm_subgoal_marker.pose.orientation.y = q[1]
                    arm_subgoal_marker.pose.orientation.z = q[2]
                    arm_subgoal_marker.pose.orientation.w = q[3]
                    t = rospy.Duration()
                    arm_subgoal_marker.lifetime = t
                    arm_subgoal_marker.scale.x = 0.5
                    arm_subgoal_marker.scale.y = 0.2
                    arm_subgoal_marker.scale.z = 0.0
                    arm_subgoal_marker.color.a = 1.0
                    arm_subgoal_marker.color.b = 1.0
                    self.subgoal_marker_publisher.publish(arm_subgoal_marker)

                global_plan_marker = Marker()
                global_plan_marker.header.frame_id = '/base_link'
                global_plan_marker.type = global_plan_marker.POINTS
                global_plan_marker.action = global_plan_marker.ADD
                global_plan_marker.pose.orientation.w = 1
                points = []
                #print('-' * 20)
                for i in range(10):
                    #print('local:', self.obs['sensor'][0, (i * 2):(i * 2 + 2)])
                    point = Point()
                    point.x = self.obs['sensor'][0, i * 2]
                    point.y = self.obs['sensor'][0, i * 2 + 1]
                    point.z = 0.0
                    points.append(point)
                global_plan_marker.points = points
                t = rospy.Duration()
                global_plan_marker.lifetime = t
                global_plan_marker.scale.x = 0.2
                global_plan_marker.scale.y = 0.2
                global_plan_marker.scale.z = 0.2
                global_plan_marker.color.a = 1.0
                global_plan_marker.color.g = 1.0
                self.global_plan_marker_publisher.publish(global_plan_marker)

                goal_marker = Marker()
                goal_marker.header.frame_id = '/base_link'
                goal_marker.type = goal_marker.POINTS
                goal_marker.action = goal_marker.ADD
                goal_marker.pose.orientation.w = 1
                point = Point()
                point.x = self.obs['sensor'][0, 20]
                point.y = self.obs['sensor'][0, 21]
                point.z = 0.0
                goal_marker.points = [point]
                t = rospy.Duration()
                goal_marker.lifetime = t
                goal_marker.scale.x = 0.4
                goal_marker.scale.y = 0.4
                goal_marker.scale.z = 0.4
                goal_marker.color.a = 1.0
                goal_marker.color.b = 1.0
                goal_marker.color.r = 1.0
                self.goal_marker_publisher.publish(goal_marker)

                print('-' * 20)
                print('using base' if use_base else 'using arm')
                if use_base:
                    print('base_subgoal_pos:', real_action[1:3])
                    print('base_subgoal_orn:', real_action[3])
                else:
                    print('arm_subgoal_pos', real_action[4:7])
                    print('interaction_vector', real_action[7:9])

                self.finished_prev_action = False
                self.policy_action_publisher.publish(policy_action_msg)
                print('publish action')
                print('-' * 20)

                """
                # rescale action from [-1, 1] to [-0.05, 0.1]
                real_action = action * 0.15 / 2.0 + 0.025
                # print('action', action)
                # print('real_action', real_action)
                vel_msg = Twist()
                vel_msg.linear.x = (real_action[0] + real_action[1]) * 1.5
                # 1.11m is the perimeter of the robot base
                vel_msg.angular.z = (real_action[1] - real_action[0]) * 1.5 * (np.pi * 2.0 / 1.11)
                print('linear:', vel_msg.linear.x)
                print('angular:', vel_msg.angular.z)
                self.velocity_publisher.publish(vel_msg)
                """
    node = InferenceROSNode()
    node.run()


if __name__ == '__main__':
    flags.mark_flag_as_required('root_dir')
    app.run(main)
