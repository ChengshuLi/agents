from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle


def get_metrics(episodes):
    success_rate = np.mean([episode.success for episode in episodes])
    collision_rate = np.mean([episode.collision for episode in episodes])
    timeout_rate = np.mean([episode.timeout for episode in episodes])
    personal_space_violation = np.mean([episode.personal_space_violation for episode in episodes])
    total_personal_space = np.mean([episode.total_personal_space for episode in episodes])
    min_personal_space = np.mean([episode.min_personal_space for episode in episodes])
    # spl = np.mean([episode.success * episode.path_efficiency for episode in episodes])
    # kinematic_disturbance = np.mean([episode.success * episode.kinematic_disturbance for episode in episodes])
    # dynamic_disturbance_a = np.mean([episode.success * episode.dynamic_disturbance_a for episode in episodes])
    # dynamic_disturbance_b = np.mean([episode.success * episode.dynamic_disturbance_b for episode in episodes])
    # collision_step = np.mean([episode.collision_step for episode in episodes])
    return {
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'timeout_rate': timeout_rate,
        'personal_space_violation': personal_space_violation,
        'average_personal_space': average_personal_space,
        'min_personal_space': min_personal_space,
        # 'spl': spl,
        # 'kinematic_disturbance': kinematic_disturbance,
        # 'dynamic_disturbance_a': dynamic_disturbance_a,
        # 'dynamic_disturbance_b': dynamic_disturbance_b,
        # 'collision_step': collision_step
    }


def save(episodes, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(episodes, f)
