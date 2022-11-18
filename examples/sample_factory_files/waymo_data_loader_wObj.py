# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Dataloader for imitation learning in Nocturne."""
from collections import defaultdict
import random
import os
import json

import torch
from pathlib import Path
import numpy as np

from cfgs.config import ERR_VAL
from nocturne import Simulation

# import sys
# import gc

# def actualsize(input_obj):
#     memory_size = 0
#     ids = set()
#     objects = [input_obj]
#     while objects:
#         new = []
#         for obj in objects:
#             if id(obj) not in ids:
#                 ids.add(id(obj))
#                 memory_size += sys.getsizeof(obj)
#                 new.append(obj)
#         objects = gc.get_referents(*new)
#     return memory_size


def _process_scenario(scenario_path, scenario_config, dataloader_config, 
                      to_gpu=False, device=None):
    tmin = dataloader_config.get('tmin', 0)
    tmax = dataloader_config.get('tmax', 90)
    view_dist = dataloader_config.get('view_dist', 80)
    view_angle = dataloader_config.get('view_angle', np.radians(120))
    dt = dataloader_config.get('dt', 0.1)
    expert_action_bounds = dataloader_config.get('expert_action_bounds',
                                                 [[-3, 3], [-0.7, 0.7]])
    expert_position = dataloader_config.get('expert_position', True)
    state_normalization = dataloader_config.get('state_normalization', 100)
    n_stacked_states = dataloader_config.get('n_stacked_states', 5)

    # create simulation
    sim = Simulation(str(scenario_path), scenario_config)
    scenario = sim.getScenario()

    # set objects to be expert-controlled
    for obj in scenario.getObjects():
        obj.expert_control = True

    # we are interested in imitating vehicles that moved
    objects_that_moved = scenario.getObjectsThatMoved()
    objects_of_interest = [
        obj for obj in scenario.getVehicles() if obj in objects_that_moved
    ]

    # initialize values if stacking states
    stacked_state = defaultdict(lambda: None)
    initial_warmup = n_stacked_states - 1

    state_list = []
    action_list = []
    obj_list = []

    # iterate over timesteps and objects of interest
    for time in range(tmin, tmax):
        objlist = []
        for obj in objects_of_interest:
            # get state
            objlist.append(obj)
            ego_state = scenario.ego_state(obj)
            visible_state = scenario.flattened_visible_state(
                obj, 
                view_dist=view_dist, 
                view_angle=view_angle, 
                head_angle=obj.head_angle)
            state = np.concatenate((ego_state, visible_state))

            # normalize state
            state /= state_normalization

            # stack state
            if n_stacked_states > 1:
                if stacked_state[obj.getID()] is None:
                    stacked_state[obj.getID()] = np.zeros(
                        len(state) * n_stacked_states, dtype=state.dtype)
                stacked_state[obj.getID()] = np.roll(
                    stacked_state[obj.getID()], len(state))
                stacked_state[obj.getID()][:len(state)] = state

            if np.isclose(obj.position.x, ERR_VAL):
                continue

            if not expert_position:
                # get expert action
                expert_action = scenario.expert_action(obj, time)
                # check for invalid action (because no value available for taking derivative)
                # or because the vehicle is at an invalid state
                if expert_action is None:
                    continue
                expert_action = expert_action.numpy()
                # now find the corresponding expert actions in the grids

                # throw out actions containing NaN or out-of-bound values
                if np.isnan(expert_action).any() \
                        or expert_action[0] < expert_action_bounds[0][0] \
                        or expert_action[0] > expert_action_bounds[0][1] \
                        or expert_action[1] < expert_action_bounds[1][0] \
                        or expert_action[1] > expert_action_bounds[1][1]:
                    continue
            else:
                expert_pos_shift = scenario.expert_pos_shift(obj, time)
                if expert_pos_shift is None:
                    continue
                expert_pos_shift = expert_pos_shift.numpy()
                expert_heading_shift = scenario.expert_heading_shift(
                    obj, time)
                if expert_heading_shift is None \
                        or expert_pos_shift[0] < expert_action_bounds[0][0] \
                        or expert_pos_shift[0] > expert_action_bounds[0][1] \
                        or expert_pos_shift[1] < expert_action_bounds[1][0] \
                        or expert_pos_shift[1] > expert_action_bounds[1][1] \
                        or expert_heading_shift < expert_action_bounds[2][0] \
                        or expert_heading_shift > expert_action_bounds[2][1]:
                    continue
                expert_action = np.concatenate(
                    (expert_pos_shift, [expert_heading_shift]))

            if stacked_state[obj.getID()] is not None:
                if initial_warmup <= 0:  # warmup to wait for stacked state to be filled up
                    state = stacked_state[obj.getID()]
                    state = torch.tensor(state, device=device) 
                    expert_action = torch.tensor(expert_action, device=device) 

                    state_list.append(state)
                    action_list.append(expert_action)
                    obj_list.append(obj.getID())
            else:
                
                state = torch.tensor(state, device=device) 
                expert_action = torch.tensor(expert_action, device=device) 

                state_list.append(state)
                action_list.append(expert_action)
                obj_list.append(obj.getID())
        #print(objlist)
        # step the simulation
        sim.step(dt)
        if initial_warmup > 0:
            initial_warmup -= 1

    return state_list, action_list, obj_list

def _get_waymo_iterator(paths, dataloader_config, scenario_config, 
                        state_only=True,
                        to_gpu=False, device=None):
    # if worker has no paths, return an empty iterator
    if len(paths) == 0:
        return

    #TODO - we need to make sure the demonstration ordering is consistent (at least within each scenario)
    state_list_all, action_list_all, obj_list_all = [], [], []
    for scenario_path in paths:
        print(scenario_path)
        state_list, action_list, obj_list =_process_scenario(scenario_path=scenario_path, 
                                                             scenario_config=scenario_config, 
                                                             dataloader_config=dataloader_config,
                                                             to_gpu=to_gpu,
                                                             device=device
                                                             )
        state_list_all += state_list
        obj_list_all += obj_list
        
        if not state_only:
            action_list_all += action_list

    if state_only:
        temp = list(zip(obj_list_all, state_list_all))
        # TEMP DEBUG: only return demos corresponding to agent 0
        temp = [(obj, state) for obj, state in temp if obj == 0]

    else:
        temp = list(zip(obj_list_all, state_list_all, action_list_all))
        # TEMP DEBUG: only return demos corresponding to agent 0
        temp = [(obj, state, act) for obj, state, act in temp if obj == 0]

    while True:
        random.shuffle(temp)
        if state_only:
            for obj_return, state_return in temp:
                yield (obj_return, state_return)
        else:
            for obj_return, state_return, action_return in temp:
                yield (obj_return, state_return, action_return)

                
class WaymoDataset(torch.utils.data.IterableDataset):
    """Waymo dataset loader."""

    def __init__(self,
                 data_path,
                 dataloader_config={},
                 scenario_config={},
                 file_limit=None,
                 to_gpu=False, 
                 device=None):
        super(WaymoDataset).__init__()
        # save configs
        self.dataloader_config = dataloader_config
        self.scenario_config = scenario_config
        
        # get paths of dataset files (up to file_limit paths)
        # self.file_paths = list(
            # Path(data_path).glob('tfrecord*.json'))[:file_limit]
        with open(os.path.join(data_path,
                               'valid_files.json')) as file:
            valid_veh_dict = json.load(file)
            self.file_paths = list(valid_veh_dict.keys())
            # sort the files so that we have a consistent order
            self.file_paths = sorted(self.file_paths)

        if file_limit != -1:
            self.file_paths = self.file_paths[0:file_limit]
        self.file_paths = [os.path.join(data_path, file_path) for file_path in self.file_paths]

        print(f'WaymoDataset: loading {len(self.file_paths)} files.')
        print(f"First filepath is {self.file_paths[0]}")

        # sort the paths for reproducibility if testing on a small set of files
        # self.file_paths.sort()
        self.to_gpu = to_gpu
        self.device = device

    def __iter__(self):
        """Partition files for each worker and return an (state, expert_action) iterable."""
        # get info on current worker process
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # single-process data loading, return the whole set of files
            return _get_waymo_iterator(self.file_paths, self.dataloader_config,
                                       self.scenario_config,
                                       state_only=True, 
                                       to_gpu=self.to_gpu, 
                                       device=self.device
                                       )

        # distribute a unique set of file paths to each worker process
        worker_file_paths = np.array_split(
            self.file_paths, worker_info.num_workers)[worker_info.id]
        return _get_waymo_iterator(list(worker_file_paths),
                                   self.dataloader_config,
                                   self.scenario_config,
                                   state_only=True,
                                   to_gpu=self.to_gpu,
                                   device=self.device
                                   )


if __name__ == '__main__':
    dataset = WaymoDataset(data_path='dataset/tf_records',
                           file_limit=20,
                           dataloader_config={
                               'view_dist': 80,
                               'n_stacked_states': 3,
                           },
                           scenario_config={
                               'start_time': 0,
                               'allow_non_vehicles': True,
                               'spawn_invalid_objects': True,
                           })

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
    )

    for i, x in zip(range(100), data_loader):
        print(i, x[0].shape, x[1].shape)
