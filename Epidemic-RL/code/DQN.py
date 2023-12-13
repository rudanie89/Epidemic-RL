from env import *
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from dataclasses import dataclass
from typing import Any
from random import sample
#import wandb
from collections import deque
import numpy as np
from itertools import count

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

env = socialEnv()
observation = env.reset()

def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t

@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool

class Model(nn.Module):
    def __init__(self, obs_shape, num_actions, lr=0.00001):
        super(Model, self).__init__()
        assert len(obs_shape) == 2, "This network only works for flat observations"
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_shape[0]*obs_shape[1], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, num_actions),
        )
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, buffer_size=2000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)

    def insert(self, sars):
        self.buffer.append(sars)

    def sample(self, num_samples):
        assert num_samples <= len(self.buffer)
        return sample(self.buffer, num_samples)

def train_step(model, state_transitions, tgt, num_actions, device, gamma):
    cur_states = torch.stack(([flatten(torch.Tensor(s.state)) for s in state_transitions])).to(device)
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions])).to(device)
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])).to(device)
    next_states = torch.stack(([flatten(torch.Tensor(s.next_state)) for s in state_transitions])).to(device)
    actions = [s.action for s in state_transitions]
    #flatten(torch.Tensor(s.next_state))

    with torch.no_grad():
        qvals_next = tgt(next_states).max(-1)[0]

    qvals = model(cur_states)
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)

    loss_fn =nn.SmoothL1Loss()
    loss = loss_fn(torch.sum(qvals * one_hot_actions, -1),rewards.squeeze() + mask[:, 0] * qvals_next*gamma)
    model.opt.zero_grad()
    loss.backward()
    ## moi them clip va dao vi tri zero_grad
    for param in model.net.parameters(): #se luu y khi dem net vao day -> ok.
        param.grad.data.clamp_(-1, 1)
    model.opt.step()
    return loss

def main(test=False, chkpt=None, device="cuda"):
    env = socialEnv()
    observation,_ = env.reset()
    dqn = Model(observation.shape, env.action_space.n).to(device)

    if chkpt is not None:
        dqn.load_state_dict(torch.load(chkpt))
        dqn.eval()

    tgt = Model(observation.shape, env.action_space.n).to(device)
    tgt.load_state_dict(dqn.state_dict())
    tgt.eval()

    tau = 100
    eps = 1
    eps_decay = 0.998
    eps_min = 0.001
    sample_size = 32

    rb = ReplayBuffer()
    gamma = 0.6

    points = []
    losspoints = []
    recoveredPoint = dict()
    action_traindict=dict()
    infected_nodesD = dict()
    infected_traindict = dict()
    snext_traindict = dict()
    inext_traindict = dict()
    qpredict=dict()

    episode_rewards = []

    if test:
        reward_test=dict()
        action_testdict=dict()

    for i_episode in range(5000):
        rolling_reward = 0
        if test:
            listReward=[]
            action_test=[]

        last_observation,_ = env.reset()
        episode_loss = 0

        if not test:
            if (i_episode+1) % tau == 0:
                tgt.load_state_dict(m.state_dict())
                torch.save(tgt.state_dict(), f"MHP/output/syntheticnew/edge002/bud20/dqnEpi/{i_episode + 1}.pth")

        recoveredEpi = []
        action_train=[]
        snext_train = []
        infected_train = []
        inext_train = []
        qpredict_i=[]

        for t in count():
            state = last_observation
            if test:
                eps = 0 #0.05
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = dqn(flatten(torch.Tensor(state)).to(device)).max(-1)[-1].item()
                print("use action from model: ", action)

            print(round((torch.max(dqn(flatten(torch.Tensor(state)).to(device))).item()),4))
            qpredict_i.append(round((torch.max(dqn(flatten(torch.Tensor(state)).to(device))).item()),4))

            #if test:
            #    qpredict_iTest.append(round((torch.max(dqn(flatten(torch.Tensor(state)).to(device))).item()), 4))

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
          
            rolling_reward += RLreward

            reward = RLreward 
            recoveredEpi.append((t+1, int(reward*100))) #, recoverednode
            action_train.append((t + 1, action))
            infected_train.append((t + 1, int(infectnext)))
            snext_train.append((t+1,int(snext)))
            inext_train.append((t+1,int(inext)))

            if test:
                listReward.append((t+1, int(reward*100))) 
                action_test.append((t+1,action))
            #if done:
            #  break

            rb.insert(Sarsd(state, action, reward, next_state, done))
            last_observation = observation

            if ((not test) and len(rb.buffer) > 32 and i_episode>=90): #
                loss = train_step(dqn, rb.sample(sample_size), tgt, env.action_space.n, device,gamma)
                episode_loss = episode_loss + float(loss.detach().cpu().item())
                if (i_episode % 2 == 0):
                    print([i_episode, t+1,eps, float(loss)])

                if eps > eps_min:
                    eps = eps * eps_decay

            if done:
                points.append((i_episode, t + 1))
                losspoints.append((i_episode, episode_loss / (t + 1)))
                episode_rewards.append(rolling_reward)
                rolling_reward = 0
                # env.reset()
                break
        #save ket qua cua training

        recoveredPoint[i_episode] = recoveredEpi
        action_traindict[i_episode] = action_train
        # recovered_traindict[i_episode] = recovered_train
        infected_nodesD[i_episode] = infectednodeM
        infected_traindict[i_episode] = infected_train
        snext_traindict[i_episode] = snext_train
        inext_traindict[i_episode] = inext_train
        qpredict[i_episode]=qpredict_i

if __name__ == '__main__':
    main()
