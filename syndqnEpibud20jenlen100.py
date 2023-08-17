from syntheticnew.ablation.synenvEpibud20jenlen100 import *
import gym
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
# env = gym.make('CartPole-v1')
#
# observation = env.reset()

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

class DQNAgent:
    def __init__(self, model):
        self.model = model

    def get_actions(self, observations):
        q_vals = self.model(observations)

        return q_vals.max(-1)[1]

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
        # self.buffer = self.buffer[-self.buffer_size:]

    def sample(self, num_samples):
        assert num_samples <= len(self.buffer)
        return sample(self.buffer, num_samples)
#def update_tgt_model(m, tgt):
#    tgt.load_state_dict(m.state_dict())

def train_step(model, state_transitions, tgt, num_actions, device, gamma = 0.6):
    cur_states = torch.stack(([flatten(torch.Tensor(s.state)) for s in state_transitions])).to(device)
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions])).to(device)
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])).to(device)
    next_states = torch.stack(([flatten(torch.Tensor(s.next_state)) for s in state_transitions])).to(device)
    actions = [s.action for s in state_transitions]
    #flatten(torch.Tensor(s.next_state))

    with torch.no_grad():
        qvals_next = tgt(next_states).max(-1)[0]

    #model.opt.zero_grad()
    qvals = model(cur_states)
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)

    loss_fn =nn.SmoothL1Loss()
    loss = loss_fn(torch.sum(qvals * one_hot_actions, -1),rewards.squeeze() + mask[:, 0] * qvals_next*0.6)

    #loss = ((rewards + mask[:, 0] * qvals_next - torch.sum(qvals * one_hot_actions, -1)) ** 2).mean()
    model.opt.zero_grad()
    loss.backward()
    ## moi them clip va dao vi tri zero_grad
    for param in model.net.parameters(): #se luu y khi dem net vao day -> ok.
        param.grad.data.clamp_(-1, 1)
    model.opt.step()
    return loss

def main(test=False, chkpt=None, device="cuda"):
    #env = gym.make('CartPole-v1')
    env = socialEnv()
    observation = env.reset()
    m = Model(observation.shape, env.action_space.n).to(device)

    if chkpt is not None:
        m.load_state_dict(torch.load(chkpt))
        m.eval()

    tgt = Model(observation.shape, env.action_space.n).to(device)
    tgt.load_state_dict(m.state_dict())
    tgt.eval()

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
        #recovered_testdict = dict()
        infected_nodesTest = dict()
        infected_testdict = dict()

        qpredictTest = dict()

    for i_episode in range(5000):
        rolling_reward = 0
        if test:
            listReward=[]
            action_test=[]
            #recovered_test = []
            infected_test = []

            qpredict_iTest = []
        last_observation = env.reset()
        episode_loss = 0

        print("initial recovered nodes and infected nodes", [last_observation[:, 1].sum(),last_observation[:, 4].sum()])
        #from Top20FBbud10randominitialbetween.Top20FBbu10randominitialbet import *

        if not test:

            if (i_episode+1) % tau == 0:
                tgt.load_state_dict(m.state_dict())
                torch.save(tgt.state_dict(), f"E:/rl/dqn1/MHP/output/syntheticnew/edge002length/reducenet100/bud20jenlen100/dqnEpi/{i_episode + 1}.pth")
        if not test:
            useBud = None
            actionUsed=[]

        recoveredEpi = []
        action_train=[]
        snext_train = []
        infected_train = []
        inext_train = []

        qpredict_i=[]

        for t in count(): #range(30): #
            #actionuse = 0
            state = last_observation
            if test:
                eps = 0 #0.05
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = m(flatten(torch.Tensor(state)).to(device)).max(-1)[-1].item()

                print("use action from model: ", action)

            print(round((torch.max(m(flatten(torch.Tensor(state)).to(device))).item()),4))
            qpredict_i.append(round((torch.max(m(flatten(torch.Tensor(state)).to(device))).item()),4))

            dem = 0
            while ((action in actionUsed) and (t<=20)):
                if random.random() < eps:
                    action = env.action_space.sample()
                else:
                    action = m(flatten(torch.Tensor(state)).to(device)).max(-1)[-1].item()
                dem = dem + 1
                if dem == 5:
                    action = env.action_space.sample()
                    print("actionused: ", action)
            print("New action: ", action)

            if test:
                qpredict_iTest.append(round((torch.max(m(flatten(torch.Tensor(state)).to(device))).item()), 4))

                    #useBud = 0
                # flatten(torch.Tensor(s.next_state))
            # if not test:
            if not test:
                if (useBud is not None and useBud <= 0):
                    action = 0
                    print("no budget", useBud)

            observation, RLreward, done,user_debunker,notin,budgetAll,infectednodeM,infectnext,snext,inext,_ = env.step(action)
            useBud = budgetAll
            #if action!=0:
            actionUsed.append(action)
            # else:
            #     observation, RLreward, Realreward, done, _ = env.step(action)

            #observation, reward, done, _ = env.step(action)
            rolling_reward += RLreward #round((1/(RLreward+0.001))*10,4)
            print("action and reward at episode " + str(i_episode) + " at step " + str(t) + " :", [action,user_debunker, notin,RLreward*100,snext,inext, done] )

            reward = RLreward #round((1/(RLreward+0.001))*10,4) #reward  # / 100.0
            #recoverednode = Realreward

            recoveredEpi.append((t+1, int(reward*100))) #, recoverednode
            # recovered_train.append((t+1,int(Realreward)))
            action_train.append((t + 1, action))
            infected_train.append((t + 1, int(infectnext)))
            snext_train.append((t+1,int(snext)))
            inext_train.append((t+1,int(inext)))

            if test:
                listReward.append((t+1, int(reward*100))) #Realreward)
                action_test.append((t+1,action))
                infected_test.append((t + 1, int(infectnext)))
                # recovered_test.append((t+1,int(Realreward)))

                # if action == 1:
                #     print("Episode " + str(i_episode) + " step " + str(t) + " : ",[action, reward] )

            if (done or (useBud<=0)):
                next_state = np.zeros((100,6)) # cai nay can sua lai
            else:
                next_state = observation

            rb.insert(Sarsd(state, action, reward, next_state, done))

            #print("recovered nodes:", recoveredPoint)

            if ((not test) and len(rb.buffer) > 32 and i_episode>=90): #
                loss = train_step(m, rb.sample(sample_size), tgt, env.action_space.n, device)
                episode_loss = episode_loss + float(loss.detach().cpu().item())
                if (i_episode % 2 == 0):
                    print([i_episode, t+1,eps, float(loss)])

                if eps > eps_min:
                    eps = eps * eps_decay


            last_observation = observation

            if (done or (useBud<=0)): #or ((t+1)==15)
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

        if test:
            reward_test[i_episode] = listReward
            action_testdict[i_episode] = action_test
            # recovered_testdict[i_episode] = recovered_test
            #print(reward_test)
            infected_nodesTest[i_episode] = infectednodeM
            infected_testdict[i_episode] = infected_test

            qpredictTest[i_episode] = qpredict_iTest

        #
        # if (not test and ((i_episode+1) % 100 == 0)):
        #     saveResult = {'points': points, 'loss': losspoints}
        #     pathEp = "E:\\rl\\dqn1\\MHP\\output\\synthetic\\network100\\trainingEpi\\" + str(i_episode+1) + "losspoints1k.txt"
        #
        #     with open(pathEp, 'w+') as fEp:
        #         json.dump(saveResult, fEp, indent=4)
        #
        #     # ResultRecovered = {'recovednode': recoveredPoint}
        #     pathEp1 = "E:\\rl\\dqn1\\MHP\\output\\synthetic\\network100\\trainingEpi\\" + str(i_episode+1) + "train_reward_1kepi.txt"
        #     with open(pathEp1, 'w+') as fEp1:
        #         json.dump(recoveredPoint, fEp1, indent=4)
        #
        #     pathAc = "E:\\rl\\dqn1\\MHP\\output\\synthetic\\network100\\trainingEpi\\" + str(i_episode+1) + "train_action_dictR1k.txt"
        #     with open(pathAc, 'w+') as fAc:
        #         json.dump(action_traindict, fAc, indent=4)
        #
        #     path_inf = "E:\\rl\\dqn1\\MHP\\output\\synthetic\\network100\\trainingEpi\\" + str(i_episode+1) + "infected_no.txt"
        #     with open(path_inf, 'w+') as f_inf:
        #         json.dump(infected_traindict, f_inf, indent=4)
        #
        #     path_infnode = "E:\\rl\\dqn1\\MHP\\output\\synthetic\\network100\\trainingEpi\\" + str(i_episode+1) + "infected_nodes.txt"
        #     with open(path_infnode, 'w+') as f_infnode:
        #         json.dump(infected_nodesD, f_infnode, indent=4)
        #
        #     path_qpredict = "E:\\rl\\dqn1\\MHP\\output\\synthetic\\network100\\trainingEpi\\" + str(i_episode+1) + "qpredict.txt"
        #     with open(path_qpredict, 'w+') as f_qpredict:
        #         json.dump(qpredict, f_qpredict, indent=4)
        #
        #
        #     path_snext = "E:\\rl\\dqn1\\MHP\\output\\synthetic\\network100\\trainingEpi\\" + str(i_episode+1) + "snext_traindict.txt"
        #     with open(path_snext, 'w+') as f_snext:
        #         json.dump(snext_traindict, f_snext, indent=4)

    if not test:
        saveResult = {'points': points, 'loss': losspoints}
        path = "E:\\rl\\dqn1\\MHP\\output\\syntheticnew\\edge002length\\reducenet100\\bud20jenlen100\\trainingEpi\\losspointsR1k.txt"

        with open(path, 'w+') as f:
            json.dump(saveResult, f, indent=4)

        # ResultRecovered = {'recovednode': recoveredPoint}
        path1 = "E:\\rl\\dqn1\\MHP\\output\\syntheticnew\\edge002length\\reducenet100\\bud20jenlen100\\trainingEpi\\train_reward_R1k.txt"
        with open(path1, 'w+') as f1:
            json.dump(recoveredPoint, f1, indent=4)
        #action_traindict
        path2 = "E:\\rl\\dqn1\\MHP\\output\\syntheticnew\\edge002length\\reducenet100\\bud20jenlen100\\trainingEpi\\train_action_dictR1k.txt"
        with open(path2, 'w+') as f2:
            json.dump(action_traindict, f2, indent=4)

        path3 = "E:\\rl\\dqn1\\MHP\\output\\syntheticnew\\edge002length\\reducenet100\\bud20jenlen100\\trainingEpi\\infected_no.txt"
        with open(path3, 'w+') as f3:
            json.dump(infected_traindict, f3, indent=4)

        path4= "E:\\rl\\dqn1\\MHP\\output\\syntheticnew\\edge002length\\reducenet100\\bud20jenlen100\\trainingEpi\\infected_nodes.txt"
        with open(path4, 'w+') as f4:
            json.dump(infected_nodesD, f4, indent=4)

        # path5 = "E:\\rl\\dqn1\\cleanCode\\output\\modelsFB1k_check2\\train_recovered_dictR1k.txt"
        # with open(path5, 'w+') as f5:
        #     json.dump(recovered_traindict, f5, indent=4)
        path_qpredictAll = "E:\\rl\\dqn1\\MHP\\output\\syntheticnew\\edge002length\\reducenet100\\bud20jenlen100\\trainingEpi\\qpredict.txt"
        with open(path_qpredictAll, 'w+') as f_qpredictAll:
            json.dump(qpredict, f_qpredictAll, indent=4)

        path_snextALL = "E:\\rl\\dqn1\\MHP\\output\\syntheticnew\\edge002length\\reducenet100\\bud20jenlen100\\trainingEpi\\snext_traindict.txt"
        with open(path_snextALL, 'w+') as f_snext:
            json.dump(snext_traindict, f_snext, indent=4)

        path_inextALL = "E:\\rl\\dqn1\\MHP\\output\\syntheticnew\\edge002length\\reducenet100\\bud20jenlen100\\trainingEpi\\inext_traindict.txt"
        with open(path_inextALL, 'w+') as f_inext:
            json.dump(inext_traindict, f_inext, indent=4)



import time

start = time.time()

if __name__ == '__main__':
    main()

end = time.time()
print("time to complete!")
print(end - start) #14765.49625 #4.1015 hours

################# testing #####################
###############################################
import time
import os
start = time.time()
# from FBbud10Top20_randominitial.FBbu10Top20 import * #

device="cuda"
env = socialEnv()
observation = env.reset()
m = Model(observation.shape, env.action_space.n).to(device)

chkpt="E:/rl/dqn1/MHP/output/syntheticnew/edge002length/reducenet100/bud20jenlen100/dqnEpi/5000.pth"
m.load_state_dict(torch.load(chkpt))
m.eval()

tgt = Model(observation.shape, env.action_space.n).to(device)
tgt.load_state_dict(m.state_dict())
tgt.eval()

from syntheticnew.ablation.synenvEpiTestbud20jenlen100 import *

reward_test = dict()
action_testdict = dict()
infected_nodesTest = dict()
infected_testdict = dict()
initiate_state =  dict()

snext_testdict = dict()
inext_testdict = dict()

envtest = socialEnvTest()

for i_episode in range(50):
    listReward = []
    action_test = []
    infected_test = []
    snext_test = []
    inext_test = []

    last_observation = envtest.reset(str(i_episode))
    # last_observation = env.reset()
    initiate_state[i_episode] = last_observation

    actionUsed = []

    for t in count():
        state = last_observation
        eps = 0
        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = m(flatten(torch.Tensor(state)).to(device)).max(-1)[-1].item()
            print("use action from model: ", action)

        dem = 0
        while action in actionUsed:
            if random.random() < 0.0005:
                action = env.action_space.sample()
            else:
                action = m(flatten(torch.Tensor(state)).to(device)).max(-1)[-1].item()
                dem = dem + 1
            if dem == 10:
                action = env.action_space.sample()
                print("actionused: ", action)
        print("New action: ", action)

        observation, RLreward, done, user_debunker, notin, budgetAll, infectednodeM, infectnext,snext,inexttest, _ = envtest.step(action)
        # observation, RLreward, done, user_debunker, notin, budgetAll, infectednodeM, infectnext, snext, _ = env.step(action)
        useBud = budgetAll
        actionUsed.append(action)

        print("action and reward at episode " + str(i_episode) + " at step " + str(t) + " :",
              [action, user_debunker, notin, RLreward, done])

        listReward.append((t + 1, int(RLreward * 100)))  # Realreward)
        action_test.append((t + 1, user_debunker))
        infected_test.append((t + 1, int(infectnext)))
        snext_test.append((t+1,int(snext)))
        inext_test.append((t+1,int(inexttest)))

        if (done or (useBud<=0)):
            next_state = np.zeros((100, 6))  # cai nay can sua lai
        else:
            next_state = observation

        last_observation = observation

        if(done or (useBud<=0)):
            break

    reward_test[i_episode] = listReward
    action_testdict[i_episode] = action_test
    infected_nodesTest[i_episode] = infectednodeM
    infected_testdict[i_episode] = infected_test

    snext_testdict[i_episode] = snext_test
    inext_testdict[i_episode] = inext_test

# outpath= 'E:\\rl\\dqn1\\twitterDQN\\MHP\\output\\synthetic\\network100\\costbetween\\edge002length\\bud10\\testEpi'
outpath= 'E:\\rl\\dqn1\\MHP\\output\\syntheticnew\\edge002length\\reducenet100\\bud20jenlen100\\testEpimu'

path11 = os.path.join(outpath,'reward_test.txt')
with open(path11, 'w+') as f11:
    json.dump(reward_test, f11, indent=4)
# action_traindict
path22 = os.path.join(outpath,'action_testdict.txt')
with open(path22, 'w+') as f22:
    json.dump(action_testdict, f22, indent=4)

path33 = os.path.join(outpath,'infected_testdict.txt')
with open(path33, 'w+') as f33:
    json.dump(infected_testdict, f33, indent=4)

path44 = os.path.join(outpath,'infected_nodes.txt')
with open(path44, 'w+') as f44:
    json.dump(infected_nodesTest, f44, indent=4)

path45 = os.path.join(outpath,'snext_testdict.txt')
with open(path45, 'w+') as f45:
    json.dump(snext_testdict, f45, indent=4)

path46 = os.path.join(outpath,'inext_testdict.txt')
with open(path46, 'w+') as f46:
    json.dump(inext_testdict, f46, indent=4)

for i in initiate_state.keys():
    np.savez(os.path.join(outpath,'initialstate\\'+str(i) + '.npz'), x=initiate_state[i])

print("End time:")
end = time.time()

print("Complete time:")
print(end - start) #232.231 seconds

######################## checking ####################
#####################################################
with open('E:\\rl\\dqn1\\MHP\\output\\syntheticnew\\edge002length\\reducenet100\\bud20jenlen100\\testEpimu\\reward_test.txt', 'r') as fBud10MHP:
    dataBud10MHP = fBud10MHP.read()
outputlogBud10MHP = json.loads(dataBud10MHP)
# len(outputlogTest['1'])

reward_Bud10MHP = dict()
for i_episode in outputlogBud10MHP.keys():
    reward_list = list()
    for k in range(len(outputlogBud10MHP[i_episode])):
        reward_list.append(outputlogBud10MHP[i_episode][k][1])
    reward_Bud10MHP[i_episode] = reward_list[-1] #round(np.mean(reward_list),4)

yBud10MHP = list(reward_Bud10MHP.values())
# xTest = list(reward_Bud10MHP.keys())

np.mean(yBud10MHP)# 17.26
np.std(yBud10MHP) # 2.6443

