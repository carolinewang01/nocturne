# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Code modified from https://github.com/marlbenchmark/on-policy
"""Runner for PPO from https://github.com/marlbenchmark/on-policy."""
from pathlib import Path
import os
import time
from itertools import chain

import hydra
import imageio
import numpy as np
import setproctitle
import torch
import wandb

from algos.ppo.sep_runner import Runner as SepRunner
from algos.ppo.env_wrappers import SubprocVecEnv, DummyVecEnv

from cfgs.config import set_display_window
from nocturne.envs.wrappers import create_ppo_env


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


def make_train_env(cfg):
    """Construct a training environment."""

    def get_env_fn(rank):
        def init_env():
            env = create_ppo_env(cfg, rank)
            # TODO(eugenevinitsky) implement this
            env.seed(cfg.seed + rank * 1000)
            return env
        return init_env

    if cfg.algorithm.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(cfg.algorithm.n_rollout_threads)])


def make_eval_env(cfg):
    """Construct an eval environment."""

    def get_env_fn(rank):

        def init_env():
            env = create_ppo_env(cfg)
            # TODO(eugenevinitsky) implement this
            env.seed(cfg.seed + rank * 1000)
            return env

        return init_env

    if cfg.algorithm.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(cfg.algorithm.n_eval_rollout_threads)])


def make_render_env(cfg):
    """Construct a rendering environment."""

    def get_env_fn(rank):

        def init_env():
            env = create_ppo_env(cfg)
            # TODO(eugenevinitsky) implement this
            env.seed(cfg.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(0)])


class NocturneSepRunner(SepRunner):
    """
    Runner class to perform training, evaluation and data collection for the Nocturne envs.

    WARNING: Assumes a shared policy.
    """

    def __init__(self, config):
        """Initialize."""
        super(NocturneSepRunner, self).__init__(config)
        self.cfg = config['cfg.algorithm']
        self.render_envs = config['render_envs']
        print("IN NOCTURNE RUNNER.INIT: ENV RESET OBS SHAPE ", self.envs.reset().shape)


    def run(self):
        """Run the training code."""
        print("IN NOCTURNE RUNNER.RUN: ENV RESET OBS SHAPE ", self.envs.reset().shape)

        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps
                       ) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(
                    step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (
                episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(self.algorithm_name, self.experiment_name,
                            episode * self.n_rollout_threads,
                            episodes * self.n_rollout_threads, total_num_steps,
                            self.num_env_steps,
                            int(total_num_steps / (end - start))))

                if self.use_wandb:
                    wandb.log({'fps': int(total_num_steps / (end - start))},
                              step=total_num_steps)
                env_infos = {}
                for agent_id in range(self.num_agents):
                    idv_rews = []
                    for info in infos:
                        if 'individual_reward' in info[agent_id].keys():
                            idv_rews.append(
                                info[agent_id]['individual_reward'])
                    agent_k = 'agent%i/individual_rewards' % agent_id
                    env_infos[agent_k] = idv_rews

                    # TODO(eugenevinitsky) this does not correctly account for the fact that there could be
                    # two episodes in the buffer
                    train_infos[agent_id]["average_episode_rewards"] = np.mean(
                        self.buffer[agent_id].rewards) * self.episode_length
                    print("average episode rewards for agent {} is {}".format(
                        agent_id,
                        train_infos[agent_id]["average_episode_rewards"]))
                    print(
                        f"maximum per step reward for agent {agent_id} is {np.max(self.buffer[agent_id].rewards)}"
                    )
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

            # save videos
            # if episode % self.cfg.render_interval == 0:
            #     self.render(total_num_steps)

    def warmup(self):
        """Initialize the buffers."""
        # reset env
        obs = self.envs.reset()

        # replay buffer
        # if self.use_centralized_V:
        #     share_obs = obs.reshape(self.n_rollout_threads, -1)
        #     share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents,
        #                                                     axis=1)
        # else:
        #     share_obs = obs

        # self.buffer.share_obs[0] = share_obs.copy()
        # self.buffer.obs[0] = obs.copy()
        share_obs = [] #  TODO: CHECEK THIS CODE!
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()


    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append( _t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                        np.array(list(obs[:, agent_id])),
                                        rnn_states[:, agent_id],
                                        rnn_states_critic[:, agent_id],
                                        actions[:, agent_id],
                                        action_log_probs[:, agent_id],
                                        values[:, agent_id],
                                        rewards[:, agent_id],
                                        masks[:, agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)  

    @torch.no_grad()
    def render(self, total_num_steps):
        """Visualize the env."""
        envs = self.render_envs

        all_frames = []
        for episode in range(self.cfg.render_episodes):
            episode_rewards = []
            obs = envs.reset()
            if self.cfg.save_gifs:
                image = envs.envs[0].render('rgb_array')
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros(
                (1, self.num_agents, self.recurrent_N, self.hidden_size),
                dtype=np.float32)
            masks = np.ones((1, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()

                # action, rnn_states = self.trainer.policy.act(
                #     np.concatenate(obs),
                #     np.concatenate(rnn_states),
                #     np.concatenate(masks),
                #     deterministic=True)
                # actions = np.array(np.split(_t2n(action), 1))
                # rnn_states = np.array(np.split(_t2n(rnn_states), 1))

                # if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                #     for i in range(envs.action_space[0].shape):
                #         uc_actions_env = np.eye(envs.action_space[0].high[i] +
                #                                 1)[actions[:, :, i]]
                #         if i == 0:
                #             actions_env = uc_actions_env
                #         else:
                #             actions_env = np.concatenate(
                #                 (actions_env, uc_actions_env), axis=2)
                # elif envs.action_space[0].__class__.__name__ == 'Discrete':
                #     actions_env = np.squeeze(
                #         np.eye(envs.action_space[0].n)[actions], 2)
                # else:
                #     raise NotImplementedError
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                   
                # [envs, agents, dim]
                actions_env = []
                # for i in range(self.n_rollout_threads):
                for i in range(1):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                # rnn_states[dones] = np.zeros(
                #     ((dones).sum(), self.recurrent_N, self.hidden_size),
                #     dtype=np.float32)
                # masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
                # masks[dones] = np.zeros(((dones).sum(), 1), dtype=np.float32)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                # masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.cfg.save_gifs:
                    image = envs.envs[0].render('rgb_array')
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.cfg.ifi:
                        time.sleep(self.cfg.ifi - elapsed)
                else:
                    envs.render('human')

                if np.all(dones[0]):
                    break

            # note, every rendered episode is exactly the same since there's no randomness in the env and our actions
            # are deterministic
            # TODO(eugenevinitsky) why is this lower than the non-render reward?
            # render_val = np.mean(np.sum(np.array(episode_rewards), axis=0))
            # print("episode reward of rendered episode is: " + str(render_val))
            # if self.use_wandb:
            #     wandb.log({'render_rew': render_val}, step=total_num_steps)
            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

        if self.cfg.save_gifs:
            if self.use_wandb:
                np_arr = np.stack(all_frames).transpose((0, 3, 1, 2))
                wandb.log({"video": wandb.Video(np_arr, fps=4, format="gif")},
                          step=total_num_steps)
            # else:
            imageio.mimsave(os.getcwd() + '/render.gif',
                            all_frames,
                            duration=self.cfg.ifi)


@hydra.main(config_path='../../cfgs/', config_name='config')
def main(cfg):
    """Run the on-policy code."""
    print("TYPE CFG", type(cfg))
    print("CFG KEYS ", cfg.keys())
    # print("CFG ALGORITH ", cfg.algorithmrithm)
    # print("CFG ALGORITHM NAME ", cfg.algorithmrithm_name)
    set_display_window()
    logdir = Path(os.getcwd())
    if cfg.wandb_id is not None:
        wandb_id = cfg.wandb_id
    else:
        wandb_id = wandb.util.generate_id()
        # with open(os.path.join(logdir, 'wandb_id.txt'), 'w+') as f:
        #     f.write(wandb_id)
    wandb_mode = "disabled" if (cfg.debug or not cfg.wandb) else "online"

    if cfg.wandb:
        run = wandb.init(config=cfg,
                         project=cfg.wandb_name,
                         name=wandb_id,
                         group='ppov2_' + cfg.experiment,
                         resume="allow",
                         settings=wandb.Settings(start_method="fork"),
                         mode=wandb_mode)
    else:
        if not logdir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [
                int(str(folder.name).split('run')[1])
                for folder in logdir.iterdir()
                if str(folder.name).startswith('run')
            ]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        logdir = logdir / curr_run
        if not logdir.exists():
            os.makedirs(str(logdir))

    if cfg.algorithm.algorithm_name == "rmappo":
        assert (cfg.algorithm.use_recurrent_policy
                or cfg.algorithm.use_naive_recurrent_policy), (
                    "check recurrent policy!")
    elif cfg.algorithm.algorithm_name == "mappo":
        assert (not cfg.algorithm.use_recurrent_policy
                and not cfg.algorithm.use_naive_recurrent_policy), (
                    "check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda
    if 'cpu' not in cfg.algorithm.device and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device(cfg.algorithm.device)
        torch.set_num_threads(cfg.algorithm.n_training_threads)
        # if cfg.algorithm.cuda_deterministic:
        #     import torch.backends.cudnn as cudnn
        #     cudnn.benchmark = False
        #     cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(cfg.algorithm.n_training_threads)

    setproctitle.setproctitle(
        str(cfg.algorithm.algorithm_name) + "-" + str(cfg.experiment))

    # seed
    torch.manual_seed(cfg.algorithm.seed)
    torch.cuda.manual_seed_all(cfg.algorithm.seed)
    np.random.seed(cfg.algorithm.seed)

    # env init
    envs = make_train_env(cfg)
    print("ENV RESET OBS SHAPE ", envs.reset().shape)
    eval_envs = make_eval_env(cfg)
    render_envs = make_render_env(cfg)
    # TODO(eugenevinitsky) hacky
    # num_agents = envs.reset().shape[1]

    # TODO: hacky; only works for a single rollout thread
    num_agents = len(envs.observation_space)

    config = {
        "cfg.algorithm": cfg.algorithm,
        "envs": envs,
        "eval_envs": eval_envs,
        "render_envs": render_envs,
        "num_agents": num_agents,
        "device": device,
        "logdir": logdir
    }

    # run experiments
    runner = NocturneSepRunner(config)
    runner.run()

    # post process
    envs.close()
    if cfg.algorithm.use_eval and eval_envs is not envs:
        eval_envs.close()

    if cfg.wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(
            str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == '__main__':
    main()
