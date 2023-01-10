#!/usr/bin/env python3
import gym
import ptan
import numpy as np
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
from gym.envs.registration import make, register, registry, spec

from lib import common
import os
from shutil import rmtree
from sheepherding import Sheepherding

GAMMA = 0.99
LEARNING_RATE = 0.00001
ENTROPY_BETA = 0.03
BATCH_SIZE = 128
NUM_ENVS = 100

REWARD_STEPS = 6
CLIP_GRAD = 0.1


class SheepherdingA2C(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(SheepherdingA2C, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)            
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


def unpack_batch(batch, net, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []                
    
    for idx, exp in enumerate(batch):                                
        states.append(np.array(exp.state, copy=False))
        actions.append(int(exp.action))
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.last_state, copy=False))
    
    states_v = torch.FloatTensor(
        np.array(states, copy=False)).to(device)
    actions_t = torch.LongTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = torch.FloatTensor(np.array(last_states, copy=False)).to(device)
        last_vals_v = net(last_states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        last_vals_np *= GAMMA ** REWARD_STEPS
        rewards_np[not_done_idx] += last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)

    return states_v, actions_t, ref_vals_v

def play_game(agent, env, folder_path):
    
    print("play_game")
    os.makedirs(folder_path)
    state = env.reset()    
    total_reward = 0
    done = False
    while not done:
        action = agent([state])[0][0]        
        state, reward, done, _ = env.step(action)
        #print("reward: "+str(reward))
        #print("total_reward: "+str(total_reward))                
        total_reward += reward    
    videofilepath = folder_path + "/sheepherding_"+str(np.round(total_reward,2)) +".gif"
    #print("play_game: folder_path: " + str(folder_path))
    #print("play_game: videofilepath: " + str(videofilepath))
    #print("play_game: os.path.exists(folder_path): " + str(os.path.exists(folder_path)))
    #print("play_game: os.path.exists(videofilepath): " + str(os.path.exists(videofilepath)))
    env.store_frames_in_gif(videofilepath)
    #print("play_game: AFTER os.path.exists(folder_path): " + str(os.path.exists(folder_path)))
    #print("play_game: AFTER os.path.exists(videofilepath): " + str(os.path.exists(videofilepath)))

if __name__ == "__main__":
    register(
        id="Sheepherding-v0",
        entry_point="sheepherding:Sheepherding",
        max_episode_steps=300,
        reward_threshold=1000000,
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    #make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))
    
    #self.env = Game(visualize=False, load_map=True, map_name=self.map_name)
    #env = Sheepherding(**strombom_typical_values)
    #play_env = gym.make("Sheepherding-v0")
    game_storing_folder = "training_games"
    if os.path.exists(game_storing_folder):
        rmtree(game_storing_folder)
    os.makedirs(game_storing_folder)
    #print("game_storing_folder: " +  str(game_storing_folder))
    #print("os.path.exists(game_storing_folder): " + str(os.path.exists(game_storing_folder)))

    #make_env = lambda: Sheepherding(**strombom_typical_values)
    stack_frames = 4
    skip_frames = 4
    make_env = lambda: ptan.common.wrappers.wrap_dqn(gym.make("Sheepherding-v0"), stack_frames=stack_frames, skip=skip_frames)
    play_env = ptan.common.wrappers.wrap_dqn(gym.make("Sheepherding-v0"), stack_frames=stack_frames, skip=skip_frames)
    envs = [make_env() for _ in range(NUM_ENVS)]
    writer = SummaryWriter(comment="-sheepherding-a2c_" + args.name)

    net = SheepherdingA2C((stack_frames,84,84), 8).to(device)
    print(net)
        
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], apply_softmax=True, device=device)
    
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS, vectorized=False)
    
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)

    batch = []      
    
    with common.RewardTracker(writer, stop_reward=np.inf) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
            for step_idx, exp in enumerate(exp_source):       

                batch.append(exp)
                #print("len(tracker.total_rewards): " + str(len(tracker.total_rewards)))
                # handle new rewards
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards:
                    if tracker.reward(new_rewards[0], step_idx):
                        break

                if len(batch) < BATCH_SIZE:
                    continue
                
                #num_of_games_trained = len(tracker.total_rewards)
                #print("num_of_games_trained: " + str(num_of_games_trained))
                #if True:                    
                ##if num_of_games_trained > 0 and num_of_games_trained % 100 == 0:                    
                #    
                #    folder_path = str(game_storing_folder)+"/"+str(num_of_games_trained)
                #    
                #    if not os.path.exists(folder_path):
                #        print("num_of_games_trained: " + str(num_of_games_trained))
                #        play_game(agent, play_env, folder_path)            
                        
                states_v, actions_t, vals_ref_v = unpack_batch(batch, net, device=device)
                batch.clear()

                optimizer.zero_grad()
                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                # calculate policy gradients only
                loss_policy_v.backward(retain_graph=True)
                grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                        for p in net.parameters()
                                        if p.grad is not None])

                # apply entropy and value gradients
                loss_v = entropy_loss_v + loss_value_v
                loss_v.backward()
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                # get full loss
                loss_v += loss_policy_v

                tb_tracker.track("advantage",       adv_v, step_idx)
                tb_tracker.track("values",          value_v, step_idx)
                tb_tracker.track("batch_rewards",   vals_ref_v, step_idx)
                tb_tracker.track("loss_entropy",    entropy_loss_v, step_idx)
                tb_tracker.track("loss_policy",     loss_policy_v, step_idx)
                tb_tracker.track("loss_value",      loss_value_v, step_idx)
                tb_tracker.track("loss_total",      loss_v, step_idx)
                tb_tracker.track("grad_l2",         np.sqrt(np.mean(np.square(grads))), step_idx)
                tb_tracker.track("grad_max",        np.max(np.abs(grads)), step_idx)
                tb_tracker.track("grad_var",        np.var(grads), step_idx)
