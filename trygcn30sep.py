from chapter5extra.epienvgcn import *
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from torch_geometric.data import Data
from torch_geometric.data import DataLoader
from torch_geometric.data import Dataset

from tqdm import tqdm
from torch_geometric.nn import GCNConv
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import global_add_pool
from torch_scatter import scatter_mean

from dataclasses import dataclass
from typing import Any
from random import sample
from collections import deque
from itertools import count

import numpy as np
import random
from itertools import permutations

from matplotlib import pyplot as plt

if torch.cuda.is_available():
    device = torch.device("cuda")

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


env = socialEnv()
observation,_ = env.reset()

@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    next_state: Any
    #edge_relat: Any
    done: bool

### define graph convolutional network ntn day? se dung scatter_mean() function
##in forward function thi minh cung de databatch, nhung cung de x, adj
# REINFROCE Network
class REINFORCE_graph(nn.Module):
    def __init__(self, state_space=None,
                 action_space=None,
                 num_hidden_layer=2,
                 hidden_dim=None,
                 learning_rate=None):

        super(REINFORCE_graph, self).__init__()
        assert state_space is not None, "None state_space input: state_space should be assigned."
        assert action_space is not None, "None action_space input: action_space should be assigned"

        if hidden_dim is None:
            hidden_dim = state_space * 2

        self.conv1 = GCNConv(6, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, action_space)
        # self.layer_norm = LayerNorm(hidden_dim)
        #self.roll_out = []
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    #def put_data(self, data):
    #    self.roll_out.append(data)

    def forward(self,x, edge_index,batch_number): ##se thu de data thoi, may cai khac minh se lay theo data
        #x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        #x = self.layer_norm(x)
        # x = global_add_pool(self.layer_norm(x), torch.LongTensor([0 for _ in range(100*2)]).to(device))
        # x = scatter_mean(x, data.batch, dim=0)
        x = scatter_mean(x, batch_number, dim=0)
        x = F.relu(self.linear(x))
        x = self.linear2(x)
        out = F.log_softmax(x, dim=1)
        return out
##################################
class ReplayBuffer:
    def __init__(self, buffer_size=2000):
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)
    def insert(self, sars):
        self.buffer.append(sars)
        # self.buffer = self.buffer[-self.buffer_size:]
    def sample(self, num_samples): ## se ko goi ham nay vao nua dau -- ok
        assert num_samples <= len(self.buffer)
        return sample(self.buffer, num_samples)
#### prepare data for Dataloader
# train_batch = BiGraphDataset(rb.buffer, edge_indexlist)
class BiGraphDataset(Dataset):
    def __init__(self, rb_buffer, edge_indexlist):
        self.rb_sample = rb_buffer
        self.listedge_index = edge_indexlist
    def __len__(self):
        return len(self.rb_sample)

    def __getitem__(self, index):
        state = self.rb_sample[index].state ##I will extract moi thanh phan o moi element of rb[index), including:
        action = self.rb_sample[index].action
        next_state = self.rb_sample[index].next_state
        reward = self.rb_sample[index].reward
        done = self.rb_sample[index].done ### lat minh se bien doi no ve dang mask luon -- khi luu -- ok
        #torch.Tensor([0]) if s.done else torch.Tensor([1])
        mask_done = None
        if done:
            mask_done = 0
        else:
            mask_done = 1
        ### minh se them edge_index o day luon -> ok
        edge_index = torch.tensor(self.listedge_index, dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        #torch.tensor(actions, dtype=torch.int64)
        return Data(x=torch.tensor(state, dtype=torch.float32),edge_index=edge_index,action=torch.tensor(action, dtype=torch.int64),
                    reward=torch.tensor(reward,dtype=torch.float32),
                    x_next=torch.tensor(next_state, dtype=torch.float),mask_done=torch.tensor(mask_done,dtype=torch.float32))


######## training ###########
##############################
#train_step(model, state_transitions, tgt, num_actions, device, gamma)
def train_step(model, data_batch, tgt, num_actions, device, gamma):
    ####phan tren nay se bo #tat ca deu da luu tren data_batch.
    cur_states = torch.stack(([flatten(torch.Tensor(s.state)) for s in state_transitions])).to(device)
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions])).to(device)
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])).to(device)
    next_states = torch.stack(([flatten(torch.Tensor(s.next_state)) for s in state_transitions])).to(device)
    actions = [s.action for s in state_transitions]
    #flatten(torch.Tensor(s.next_state))
    # s_g = create_torch_graph_data(s, edgelist_index)
    # a_prob = Policy(s_g.x.to(device), s_g.edge_index.to(device))

    with torch.no_grad():
        # qvals_next = tgt(next_states).max(-1)[0]
        qvals_next = tgt(data_batch.next_states,data_batch.edge_index,data_batch.batch).max(-1)[0]

    #model.opt.zero_grad()
    # qvals = model(cur_states)
    qvals = model(data_batch.state,data_batch.edge_index,data_batch.batch)
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device) #actions = data_batch.actions
    loss_fn =nn.SmoothL1Loss()
    loss = loss_fn(torch.sum(qvals * one_hot_actions, -1),rewards.squeeze() + mask[:, 0] * qvals_next*gamma) #rewards = data_batch.reward
    #loss = ((rewards + mask[:, 0] * qvals_next - torch.sum(qvals * one_hot_actions, -1)) ** 2).mean()
    model.opt.zero_grad()
    loss.backward()
    ## moi them clip va dao vi tri zero_grad
    for param in model.net.parameters(): #se luu y khi dem net vao day -> ok.
        param.grad.data.clamp_(-1, 1)
    model.opt.step()
    return loss

##### thu cai nay thoi, khi load data with Dataloader xem co thuc hien duoc ko
def create_torch_graph_data(data,listedge_index):
    edge_index = torch.tensor(listedge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    node_feature = torch.tensor(data, dtype=torch.float)
    data = Data(x=node_feature, edge_index=edge_index)
    return data

observation,listedge_index = env.reset()
learning_rate = 0.0001

Policy = REINFORCE_graph(state_space=observation.shape[0],
                         action_space=env.action_space.n,
                         num_hidden_layer=0,
                         hidden_dim=128,
                         learning_rate=learning_rate).to(device)

tgt = REINFORCE_graph(state_space=observation.shape[0],
                      action_space=env.action_space.n,
                      num_hidden_layer=0,
                      hidden_dim=128,
                      learning_rate=learning_rate).to(device)
tgt.load_state_dict(Policy.state_dict())
tgt.eval()

tryqvalue = Policy(data_batch.state,data_batch.edge_index,data_batch.batch)

datatry1 = create_torch_graph_data(observation,listedge_index) ##hinh nhu cho nay ko duoc
datatry1.batch = torch.tensor([0] * datatry1.num_nodes, dtype=torch.long)
tryqvalue = Policy(datatry1.to(device))  #### the con luc  dung next-state thi sao? --- doi duoc roi, gio thu
### voi cai kia xem ntn


tryqvalue = Policy(datatry1.x.to(device),datatry1.edge_index.to(device),datatry1.batch.to(device))


########## Now it is time to check with DataLoader, first create the data for rb
tau = 100
eps = 1
eps_decay = 0.998
eps_min = 0.001
sample_size = 32

rb = ReplayBuffer()
# discount=0.99 # already give the gamma in train_step()

points = []
losspoints = []
recoveredPoint = dict()
action_traindict = dict()
infected_nodesD = dict()
infected_traindict = dict()
snext_traindict = dict()
inext_traindict = dict()

qpredict = dict()

episode_rewards = []

for i_episode in range(50):
    rolling_reward = 0
    last_observation,listedge_index = env.reset()
    # episode_loss = 0

    print("initial recovered nodes and infected nodes", [last_observation[:, 1].sum(), last_observation[:, 4].sum()])
    # if not test:
    #
    #     if (i_episode + 1) % tau == 0:
    #         tgt.load_state_dict(m.state_dict())
    #         torch.save(tgt.state_dict(),
    #                    f"E:/rl/dqn1/MHP/output/syntheticnew/edge002length/reducenet100/bud20jenlen100/dqnEpi/{i_episode + 1}.pth")
    # if not test:
    useBud = None
    actionUsed = []

    recoveredEpi = []
    action_train = []
    snext_train = []
    infected_train = []
    inext_train = []

    qpredict_i = []

    for t in count():
        state = last_observation
        action = env.action_space.sample()
        # if random.random() < eps:
        #     action = env.action_space.sample()
        # else:
        #     action = m(flatten(torch.Tensor(state)).to(device)).max(-1)[-1].item()
        #
        #     print("use action from model: ", action)
        #
        # print(round((torch.max(m(flatten(torch.Tensor(state)).to(device))).item()), 4))
        # qpredict_i.append(round((torch.max(m(flatten(torch.Tensor(state)).to(device))).item()), 4))


        if (useBud is not None and useBud <= 0):
            action = 0
            print("no budget", useBud)

        observation, RLreward, done, user_debunker, notin, budgetAll, infectednodeM, infectnext, snext, inext, _ = env.step(action)
        useBud = budgetAll
        actionUsed.append(action)
        rolling_reward += RLreward
        print("action and reward at episode " + str(i_episode) + " at step " + str(t) + " :",
              [action, user_debunker, notin, RLreward * 100, snext, inext, done])

        reward = RLreward
        if (done or (useBud <= 0)):
            next_state = np.zeros((100, 6))  # cai nay can sua lai
            break
        else:
            next_state = observation

        rb.insert(Sarsd(state, action, reward, next_state, done))
        last_observation = observation

        recoveredEpi.append((t + 1, int(reward * 100)))
        action_train.append((t + 1, action))
        infected_train.append((t + 1, int(infectnext)))
        snext_train.append((t + 1, int(snext)))
        inext_train.append((t + 1, int(inext)))



train_batch = BiGraphDataset(rb.buffer, listedge_index)
train_loader = DataLoader(train_batch, batch_size=2, shuffle=True, num_workers=0)
tqdm_train_loader = tqdm(train_loader)
dem = 0
for Batch_data in tqdm_train_loader:
    Batch_data.to(device)
    dem=dem + 1
    if dem>=3:
        break
    print(Batch_data)


tryqvalue2 = Policy(Batch_data.x.to(device),Batch_data.edge_index.to(device),Batch_data.batch.to(device))
# a_prob = Policy(Batch_data.x, Batch_data.edge_index)
qvals = Policy(Batch_data.x.to(device),Batch_data.edge_index.to(device),Batch_data.batch.to(device))
qvals_next = tgt(Batch_data.x_next.to(device),Batch_data.edge_index.to(device),Batch_data.batch.to(device)).max(-1)[0]

with torch.no_grad():
    # qvals_next = tgt(next_states).max(-1)[0]
    qvals_next = tgt(data_batch.next_states, data_batch.edge_index, data_batch.batch).max(-1)[0]

# model.opt.zero_grad()
# qvals = model(cur_states)
qvals = model(data_batch.state, data_batch.edge_index, data_batch.batch)
#one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)  # actions = data_batch.actions

one_hot_actions = F.one_hot(torch.LongTensor(Batch_data.action.cpu()), 30).to(device)

loss_fn = nn.SmoothL1Loss()
loss = loss_fn(torch.sum(qvals * one_hot_actions, -1),reward_batch2 + Batch_data.mask_done*qvals_next*0.7)


# loss = loss_fn(torch.sum(qvals * one_hot_actions, -1),
#                rewards.squeeze() + mask[:, 0] * qvals_next * gamma)  # rewards = data_batch.reward
##Batch_data.mask_done*qvals_next*0.7 # tensor([-2.3152, -2.3157], device='cuda:0', grad_fn=<MulBackward0>)
## torch.sum(qvals * one_hot_actions, -1)  #tensor([-3.3209, -3.3212], device='cuda:0', grad_fn=<SumBackward1>)