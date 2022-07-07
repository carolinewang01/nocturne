# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Wrappers and env constructors for the environments."""
from gym.spaces import Box, Discrete
import numpy as np

from nocturne.envs import BaseEnv


class OnPolicyPPOWrapper(object):
    """Wrapper to make env compatible with On-Policy code."""

    def __init__(self, env, use_images=False):
        """Wrap with appropriate observation spaces and make fixed length.

        Args
        ----
            env ([type]): [description]
            no_img_concat (bool, optional): If true, we don't concat images into the 'state' key
        """
        self._env = env
        self.use_images = use_images

        # self.action_discretization = 5
        # self.accel_grid = np.linspace(-1, 1, self.action_discretization)
        # self.steering_grid = np.linspace(-.4, .4, self.action_discretization)
        self.accel_discretization = self._env.accel_discretization
        self.steering_discretization = self._env.steering_discretization
        self.head_angle_discretization = self._env.head_angle_discretization

        self.accel_grid = self._env.accel_grid
        self.steering_grid = self._env.steering_grid
        self.head_angle_grid = self._env.head_angle_grid

        self._env.reset() # compute controlled vehicles
        self.n = len(self._env.controlled_vehicles)
        print("NUM ENV CTRLLED VEHICLES ARE ", self.n)
        print("NUM EXPERT CTRLLED VEHICLES ARE ", len(self._env.expert_controlled_vehicles))
        print("ALL VEHICLE IDES ", len(self._env.all_vehicle_ids))
        print(("SINGLE AGENT MODE "), self._env.single_agent_mode)
        print("ENV VEHICLES ", self._env.__dict__.keys())
        obs_dict = self.reset()
        # tracker used to match observations to actions
        self.agent_ids = []
        self.feature_shape = obs_dict[0].shape
        self.share_observation_space = [
            Box(low=-np.inf,
                high=+np.inf,
                shape=self.feature_shape,
                dtype=np.float32) for _ in range(self.n)
        ]

    @property
    def observation_space(self):
        """See superclass."""
        return [
            # self._env.observation_space
            Box(low=-np.inf, high=np.inf, shape=self.feature_shape)
            for _ in range(self.n)
        ]

    @property
    def action_space(self):
        """See superclass."""
        return [Discrete(self.accel_discretization * self.steering_discretization * self.head_angle_discretization) for _ in range(self.n)]

        # return [Discrete(self.action_discretization**2) for _ in range(self.n)]
        # return [self._env.action_space for _ in range(self.n)]

    def step(self, actions):
        """Convert returned dicts to lists."""
        agent_actions = {}
        for action_vec, agent_id in zip(actions, self.agent_ids):
            # during training this is a one-hot vector, during eval this is the argmax
            if action_vec.shape[0] != 1:
                action = np.argmax(action_vec)
            else:
                action = action_vec[0]
            # accel_action = self.accel_grid[int(action //
            #                                    self.action_discretization)]
            # steering_action = self.steering_grid[action %
            #                                      self.action_discretization]
            # agent_actions[agent_id] = {
            #     'accel': accel_action,
            #     'turn': steering_action
            # }
            agent_actions[agent_id] = action
            
        next_obses, rew, done, info = self._env.step(agent_actions)
        obs_n = []
        rew_n = []
        done_n = []
        info_n = []
        # TODO(eugenevinitsky) I'm a little worried that there's going to be an order mismatch here
        for key in self.agent_ids:
            if isinstance(next_obses[key], dict):
                obs_n.append(next_obses[key]['features'])
            else:
                obs_n.append(next_obses[key])
            rew_n.append([rew[key]])
            done_n.append(done[key])
            agent_info = info[key]
            agent_info['individual_reward'] = rew[key]
            info_n.append(agent_info)
        return obs_n, rew_n, done_n, info_n

    def reset(self):
        """Convert observation dict to list."""
        obses = self._env.reset()
        obs_n = []
        self.agent_ids = []
        for key in obses.keys():
            self.agent_ids.append(key)
            if not hasattr(self, 'agent_key'):
                self.agent_key = key
            if isinstance(obses[key], dict):                                                                    
                obs_n.append(obses[key]['features'])
            else:
                obs_n.append(obses[key])
        return obs_n

    def render(self, mode=None):
        """See superclass."""
        return self._env.render(mode)

    def seed(self, seed=None):
        """See superclass."""
        self._env.seed(seed)

    def __getattr__(self, name):
        """See superclass."""
        return getattr(self._env, name)


def create_env(cfg):
    """Return the base environment."""
    env = BaseEnv(cfg)
    return env


def create_ppo_env(cfg, rank=0):
    """Return a PPO wrapped environment."""
    print("\n")
    print("CREATING PPO ENV")
    env = BaseEnv(cfg, rank=rank)
    return OnPolicyPPOWrapper(env, use_images=cfg.img_as_state)
