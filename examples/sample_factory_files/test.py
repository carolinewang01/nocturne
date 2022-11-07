# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Imitation learning training script (behavioral cloning)."""
from datetime import datetime
from pathlib import Path
import pickle
import random
import json

import hydra
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from examples.sample_factory_files.waymo_data_loader_wObj import WaymoDataset
# from examples.imitation_learning.waymo_data_loader_wObj import WaymoDataset
import time


def set_seed_everywhere(seed):
    """Ensure determinism."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="../../cfgs/", config_name="config")
def main(cfg):
    """Train an IL model."""

    # put it into a namespace so sample factory code runs correctly
    class Bunch(object):

        def __init__(self, adict):
            self.__dict__.update(adict)

    cfg = Bunch(cfg)

    set_seed_everywhere(cfg.seed)
    # create dataset and dataloader
    expert_bounds = [[-6, 6], [-0.7, 0.7]] # TODO: why is this only giving 2 set of lower/upper bounds?

    dataloader_cfg = {
        'tmin': 0,
        'tmax': cfg.episode_length,# 90
        'view_dist': cfg.subscriber['view_dist'],
        'view_angle': cfg.subscriber['view_angle'],
        'dt': cfg.dt,
        'expert_action_bounds': expert_bounds, # TODO: figure out what this should be!
        'expert_position': False, # taken from actions_are_positions # TODO: figure out what this does!
        'state_normalization': 100, # TODO: check how nocturne env does state normalization!
        'n_stacked_states': cfg.subscriber['n_frames_stacked'],
    }
    scenario_cfg = dict(cfg.scenario)
    
    dataset = WaymoDataset(
        data_path=cfg.scenario_path,
        file_limit=100, # cfg.num_files,
        dataloader_config=dataloader_cfg,
        scenario_config=scenario_cfg,
        to_gpu=True,
        device="cuda:0"

    )
    data_loader = iter(
        DataLoader(
            dataset,
            batch_size=512,
            num_workers=0,
            pin_memory=False,
        ))

    obj, sample_state = next(data_loader)
    '''
    With new dataloader: 
    100 files
    Number of state-action samples: 45k
    States size: 2431733328 bytes, 2.43 gB
    Actions size 5249268 bytes, 0.005 gB
    '''
    import sys; sys.exit(0)
    start = time.time()
    for i in range(100):
        print("GENERATING NEXT EXAMPLE")
        obj, sample_state, sample_action = next(data_loader)
    end = time.time()
    print("AVG TIME TO SAMPLE BATCH ", (end - start)/100) 
    '''
     .30 seconds for old dataloader 
     .023 seconds for new dataloader
    '''
    print(sample_state.shape, sample_action.shape)

if __name__ == '__main__':
    config_path="../../cfgs/config.yaml"
    main()
