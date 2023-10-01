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
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

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

    with torch.no_grad():
        qvals_next = tgt(data_batch.x_next.to(device), data_batch.edge_index.to(device), data_batch.batch.to(device)).max(-1)[0]
        #print("qvals_next:",qvals_next)

    qvals = model(data_batch.x.to(device), data_batch.edge_index.to(device), data_batch.batch.to(device))
    #print("qvals:",qvals)
    one_hot_actions = F.one_hot(torch.LongTensor(data_batch.action.cpu()), num_actions).to(device)
    #print("one hot:",one_hot_actions)

    #print("reward:",data_batch.reward.to(device))
    #print("mask:",data_batch.mask_done.to(device))

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(torch.sum(qvals * one_hot_actions, -1), data_batch.reward.to(device) + data_batch.mask_done.to(device)* qvals_next * gamma)
    model.optimizer.zero_grad()
    loss.backward()
    ## moi them clip va dao vi tri zero_grad
    # for param in model.parameters(): #se luu y khi dem net vao day -> ok.
    #     param.grad.data.clamp_(-1, 1)
    model.optimizer.step()
    return loss

# for param in Policy.parameters(): #se luu y khi dem net vao day -> ok.
#     param.grad.data.clamp_(-1, 1)
##### thu cai nay thoi, khi load data with Dataloader xem co thuc hien duoc ko
def create_torch_graph_data(data,listedge_index):
    edge_index = torch.tensor(listedge_index, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    node_feature = torch.tensor(data, dtype=torch.float)
    data = Data(x=node_feature, edge_index=edge_index)
    return data

################# starting ###################
##############################################
device="cuda"
observation,listedge_index = env.reset()
learning_rate = 0.0001
tau = 100
eps = 1
eps_decay = 0.998
eps_min = 0.001
batch_size = 32
gamma = 0.7

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

rb = ReplayBuffer()

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

for i_episode in range(1000):
    rolling_reward = 0
    last_observation,listedge_index = env.reset()
    episode_loss = 0

    print("initial recovered nodes and infected nodes", [last_observation[:, 1].sum(), last_observation[:, 4].sum()])
    # if not test:
    #
    if (i_episode + 1) % tau == 0:
        tgt.load_state_dict(Policy.state_dict())
        torch.save(tgt.state_dict(),f"E:/pheme/code/timegcn/chapter5extra/output100nodes/dqnEpi/{i_episode + 1}.pth")
    # if not test:
    useBud = None
    actionUsed = []
    recoveredEpi = []
    action_train = []
    snext_train = []
    infected_train = []
    inext_train = []
    qpredict_i = []
    #t = 0

    #while not done:
    actionUsed = []
    for t in count():
        state = last_observation
        # action = env.action_space.sample()
        if random.random() < eps:
            action = env.action_space.sample()
            print(action)
        else:
            # tryqvalue = Policy(data_batch.state, data_batch.edge_index, data_batch.batch)
            s = create_torch_graph_data(state, listedge_index)  ##hinh nhu cho nay ko duoc
            s.batch = torch.tensor([0] * s.num_nodes, dtype=torch.long)
            #tryqvalue = Policy(datatry1.to(device))
            a_prob = Policy(s.x.to(device), s.edge_index.to(device), s.batch.to(device))
            a_distrib = Categorical(torch.exp(a_prob))
            action = a_distrib.sample()
            action = action.cpu()
            action = action[-1].item()
            print("use action from model: ", action)
        #
        # print(round((torch.max(m(flatten(torch.Tensor(state)).to(device))).item()), 4))
        s_state = create_torch_graph_data(state, listedge_index)
        s_state.batch = torch.tensor([0] * s_state.num_nodes, dtype=torch.long)
        qpredict_i.append(round((torch.max(Policy(s_state.x.to(device), s_state.edge_index.to(device), s_state.batch.to(device))).item()), 4))

        dem = 0
        actionUsed.append(action)
        while ((action in actionUsed) and (t <= 20)):
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                s = create_torch_graph_data(state, listedge_index)
                s.batch = torch.tensor([0] * s.num_nodes, dtype=torch.long)
                a_prob = Policy(s.x.to(device), s.edge_index.to(device), s.batch.to(device))
                a_distrib = Categorical(torch.exp(a_prob))
                action = a_distrib.sample()
                action = action.cpu()
                action = action[-1].item()

            dem = dem + 1
            if dem == 10:
                action = env.action_space.sample()
                print("actionused: ", action)
        print("New action: ", action)

        # if (useBud is not None and useBud <= 0):
        #     action = 0
        #     print("no budget", useBud)

        observation, RLreward, done, user_debunker, notin, budgetAll, infectednodeM, infectnext, snext, inext, _ = env.step(action)
        actionUsed.append(action)
        rolling_reward += RLreward
        #t = t +1
        print("action and reward at episode:" + str(i_episode) + " at step "  + str(t) + " :", [action, user_debunker, notin, RLreward * 100, snext, inext, done])

        reward = RLreward
        if (done or (budgetAll <= 0)):
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

        if ((len(rb.buffer) > 32) and (i_episode >= 50)):  #
            train_batch = BiGraphDataset(rb.buffer, listedge_index)
            train_loader = DataLoader(train_batch, batch_size=32, shuffle=True, num_workers=0)
            tqdm_train_loader = tqdm(train_loader)
            for Batch_data in tqdm_train_loader:
                Batch_data.to(device)
                #print("Batch_data:",Batch_data)
                loss = train_step(Policy, Batch_data, tgt, env.action_space.n, device,gamma)
                #print("loss:",loss)
                ## chac phai lay average loss ####
                episode_loss = episode_loss + float(loss.detach().cpu().item())
            if (i_episode % 2 == 0):
                print([i_episode, t + 1, eps, float(loss)])

            if eps > eps_min:
                eps = eps * eps_decay
        points.append((i_episode, t + 1))
        losspoints.append((i_episode, episode_loss / (t + 1)))
    recoveredPoint[i_episode] = recoveredEpi
    action_traindict[i_episode] = action_train
    # recovered_traindict[i_episode] = recovered_train
    infected_nodesD[i_episode] = infectednodeM
    infected_traindict[i_episode] = infected_train
    snext_traindict[i_episode] = snext_train
    inext_traindict[i_episode] = inext_train
    qpredict[i_episode] = qpredict_i

saveResult = {'points': points, 'loss': losspoints}
path = "E:\\pheme\\code\\timegcn\\chapter5extra\\output100nodes\\train\\losspointsR1k.txt"

with open(path, 'w+') as f:
    json.dump(saveResult, f, indent=4)

# ResultRecovered = {'recovednode': recoveredPoint}
path1 = "E:\\pheme\\code\\timegcn\\chapter5extra\\output100nodes\\train\\train_reward_R1k.txt"
with open(path1, 'w+') as f1:
    json.dump(recoveredPoint, f1, indent=4)
# action_traindict
path2 = "E:\\pheme\\code\\timegcn\\chapter5extra\\output100nodes\\train\\train_action_dictR1k.txt"
with open(path2, 'w+') as f2:
    json.dump(action_traindict, f2, indent=4)

path3 = "E:\\pheme\\code\\timegcn\\chapter5extra\\output100nodes\\train\\infected_no.txt"
with open(path3, 'w+') as f3:
    json.dump(infected_traindict, f3, indent=4)

path4 = "E:\\pheme\\code\\timegcn\\chapter5extra\\output100nodes\\train\\infected_nodes.txt"
with open(path4, 'w+') as f4:
    json.dump(infected_nodesD, f4, indent=4)

# path5 = "E:\\rl\\dqn1\\cleanCode\\output\\modelsFB1k_check2\\train_recovered_dictR1k.txt"
# with open(path5, 'w+') as f5:
#     json.dump(recovered_traindict, f5, indent=4)
path_qpredictAll = "E:\\pheme\\code\\timegcn\\chapter5extra\\output100nodes\\train\\qpredict.txt"
with open(path_qpredictAll, 'w+') as f_qpredictAll:
    json.dump(qpredict, f_qpredictAll, indent=4)

path_snextALL = "E:\\pheme\\code\\timegcn\\chapter5extra\\output100nodes\\train\\snext_traindict.txt"
with open(path_snextALL, 'w+') as f_snext:
    json.dump(snext_traindict, f_snext, indent=4)

path_inextALL = "E:\\pheme\\code\\timegcn\\chapter5extra\\output100nodes\\train\\inext_traindict.txt"
with open(path_inextALL, 'w+') as f_inext:
    json.dump(inext_traindict, f_inext, indent=4)


##### try to test on the same data reported