import numpy as np
import math
import networkx as nx
# from MHP.code.MHP import MHP

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import numpy as np
import random
import os
import json

from collections import Counter
from itertools import count
epsilon = 0.001

# G = nx.read_edgelist("E:\\rl\\dqn1\\datafb\\facebook_combined.txt", create_using = nx.Graph(), nodetype=int)
G=nx.generators.random_graphs.binomial_graph(100,0.02,seed=246,directed=False)

len(list(G.edges()))
#G = nx.Graph()
#G.add_edges_from([(1, 2), (2, 3)])

nbrs = list(nx.all_neighbors(G, 1))
nbrs

edgeAll = list(G.edges())
len(edgeAll)

listnodesFB = list(G.nodes)

##############
fbnode = list(G.nodes)

def areSame(A,B):
    C=np.abs(A-B)
    d=C<=epsilon
    count_d=len(C[d])
    if count_d >= 9000: #let's relax this to make sure not picking a lot of do nothing; before was: 24230
        return 0 #
    return 1

### indegree neibours
indegreeDict = {}
for i in G.nodes:
    indegreeDict[i] = [tup[0] for tup in edgeAll if tup[1] == i]
##############
outdegreeDict = {}
for i in G.nodes: #G.nodes:
    neigbor2 = [tup[1] for tup in edgeAll if tup[0] == i]
    outdegreeDict[i] =len(neigbor2)
#
# outdegreeDict[100]=0


neigDict = {}
for i in G.nodes:
    neigbor1 = list(G.neighbors(i))
    neigDict[i] = len(neigbor1)

def normalizeNei(x, maxNei,minNei):
    return 1 + (x - minNei) * (3.0 - 1) / (maxNei - minNei)

## calculate betweenness
betCent = nx.betweenness_centrality(G, normalized=True, endpoints=True)
print(type(betCent))
betCent[10]

### cost
betCentvalue = list(betCent.values())
costu = {}
for i in G.nodes:
    costu[i]=normalizeNei(betCent[i], max(betCentvalue),min(betCentvalue))


top30list=[]
for k in dict(Counter(betCent).most_common(30)).keys():
    top30list.append(k)
#top20list = [15, 1, 3, 30, 5, 71, 19, 68, 24, 2, 59, 38, 86, 84, 64, 8, 35, 90, 13, 83]
##directed graph ##[65, 71, 90, 99, 89, 60, 23, 63, 3, 49, 17, 85, 58, 56, 19, 6, 1, 66, 24, 79]


'''
Creating Env
'''
class socialEnv(gym.Env): #object):
    def __init__(self):
        super(socialEnv, self).__init__()
        self.G=G
        #self.adjG=adjG
        self.edgeAll=edgeAll
        self.infectednode = None #random.sample(initial_infected, 2)

        self.debunkersU = []
        self.mitigators = top30list #top20list

        self.muR = None #etaR
        self.muT = None #etaT
        self.action_space = spaces.Discrete(30)
        # self.observation_space = Box(low=0.0, high=1.0, shape=(600,), dtype=np.float64)
        #np.zeros((len(self.G.nodes), 6)
        self.observation_space = spaces.Box(low=np.zeros((len(self.G.nodes), 6)).reshape(-1), high=np.ones((len(self.G.nodes), 6)).reshape(-1), shape=(600,), dtype=np.float64)
        # self.current_state = None
        self.debunkers = None

        # self.budgetAll=20
        # self.dem_inf = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.mitigators = top30list #top20list

        self.infectednode = random.sample([ui for ui in list(self.G.nodes) if ui not in self.mitigators], 5)
        self.ini_par_prob = np.zeros((len(self.G.nodes), 6), dtype=np.float64)

        for i in self.G.nodes:
            if i in self.infectednode:
                prob_ag = np.array(
                    [0.00, 0.00, 0.0, 0.00, 1, 0], dtype=np.float64)
            else:
                prob_random = np.random.uniform(low=0, high=0.001, size=(2,))

                prob_ag = np.array([1 - (round(prob_random[0], 4) + round(prob_random[1], 4)), round(prob_random[0], 4), 0,
                                    round(prob_random[0]*0.6, 4), round(prob_random[1], 4), round(prob_random[1]*0.1, 4)], dtype=np.float64)
                # prob_ag = np.array([1.0, 0.0, 0.0, 0.0,
                #                     0.0, 0], dtype=np.float64)

            self.ini_par_prob[i] = prob_ag

       #self.current_state = self.ini_par_prob

        self.muR = dict()
        self.muT = dict()

        for i in self.G.nodes:
            if len(indegreeDict[i]) == 0:
                self.muR[i] = []
                self.muT[i] = []
            else:
                if i in self.infectednode:
                    self.muR[i] = []
                    self.muT[i] = []
                else:
                    nodeI = [random.uniform(0.7, 1) if p in self.infectednode else random.uniform(0.0, 0.4) for p in indegreeDict[i]]
                    self.muR[i] = dict(zip(indegreeDict[i], nodeI))

                    nodeT = [0.0 if p in self.infectednode else random.uniform(0.0, 0.4) for p in indegreeDict[i]]
                    self.muT[i] = dict(zip(indegreeDict[i], nodeT))

        self.current_state, snextIni, rnextIni, InextIni, \
        reward6Ini, reward7Ini, reward8Ini, current_listPIni = self.simulation_propagationOvertime(self.ini_par_prob,[],25,'True')

        self.budgetAll = 20
        self.dem_inf = 0

        #self.current_state=self.ini_par_prob


        return self.current_state.reshape(-1),{} #,self.edgeAll

    def step(self, action):

        # if action == 0:
        #     self.debunkers = 100
        #
        #     current_par_provM, snext, rnext, Inext, \
        #     reward6, reward7, reward8, current_listP = self.simulation_propagationOvertime(self.current_state,[],25,'True')
        #
        #     sim_AB = areSame(self.current_state, current_par_provM)
        #     self.budgetAll = self.budgetAll+0
        #     notinDe = 'nothing'

        # else:
        self.debunkers = self.mitigators[action] #self.mitigators[action-1]

        current_par_provM, snext, rnext, Inext, \
        reward6, reward7, reward8, current_listP = self.simulation_propagationOvertime(self.current_state, [self.debunkers],25,'False')
        sim_AB = areSame(self.current_state, current_par_provM)
        self.budgetAll = self.budgetAll - 1*costu[self.debunkers] #*probdictT[self.debunkers]
        notinDe = 'inDebunk'

        #done = (self.budgetAll<=0) #bool(sim_AB == 0)
        terminated = bool(self.budgetAll<=0)
        truncated = False

        if self.budgetAll>0:
            RLreward = round((rnext/100),4) #round((rnext/100)*math.exp(-outdegreeDict[self.debunkers]/100), 4) # round(rnext/100,4)
            #round((rnext/100)*m.exp(-(outdegreeDict[self.debunkers]*100)/100), 4)
        else:
            RLreward = round((rnext/100),4) #round((rnext/100)*math.exp(-outdegreeDict[self.debunkers]/100), 4) #round(rnext/100,4)

        self.current_state = current_par_provM
        self.current_I = Inext

        self.current_R = rnext

        return (self.current_state.reshape(-1), round(RLreward,4), terminated,truncated,{},)
    #self.current_state, round(RLreward,4), done,self.debunkers,notinDe,self.budgetAll,self.infectednode,round(Inext,4),round(snext,4),round(Inext,4),{}

    def simulation_propagation(self,prev_par_prob,debunkers,Timeinjectne):

        current_par_prov = np.zeros((len(self.G.nodes), 6), dtype=np.float64)

        for i in self.G.nodes:
            if len(indegreeDict[i])==0:
                current_par_prov[i] = prev_par_prob[i]
                #giu nguyen
            else:
                if i in self.infectednode: #giu nguyen
                    current_par_prov[i] = prev_par_prob[i]
                elif (len(debunkers) != 0 and (i in debunkers)):
                    current_par_prov[i] = np.array([0, 1, 0, 1, 0, 0], dtype=np.float64)
                else:
                    prob_agPre = prev_par_prob[i]

                    negAll = np.prod([1 if k in debunkers else (1 - self.muR[i][k] * prev_par_prob[k][4]) for k in indegreeDict[i]])
                    posAll = np.prod([round(1 - (random.uniform(0.8, 1)) * prev_par_prob[k][3],4) if k in debunkers else (1 - self.muT[i][k] * prev_par_prob[k][3]) for k in indegreeDict[i]])

                    # v = (1 - negAll) * posAll
                    if Timeinjectne=='True':
                        r = 0
                        v = (1 - negAll)
                    else:
                        v = (1 - negAll) * posAll
                        r = 1 - posAll
                    probSus_curr = round((1 - v - r) * prob_agPre[0],4)

                    probRec_curr = round(r * (1 - prob_agPre[1]) + prob_agPre[1], 4)
                    probDef_curr = round((1 - r) * prob_agPre[2], 4)  # self.prob_ag['def']
                    probCon_curr = round(prob_agPre[0] * v,4)  # max(prob_agPre[0] * v,1- (probSus_curr+probRec_curr)) #them vao
                    probAct_curr = round((1 - prob_agPre[1]) * r, 4)

                    probMis_curr = round((prob_agPre[4] + prob_agPre[5]) * (1 - r), 4)

                    prob_current = np.array(
                        [probSus_curr, probRec_curr, probDef_curr, probAct_curr, probCon_curr, probMis_curr])
                    current_par_prov[i] = prob_current

        snext = current_par_prov[:, 0].sum()
        rnext = current_par_prov[:, 1].sum()
        connext = current_par_prov[:, 4].sum()
        misnext = current_par_prov[:, 5].sum()

        dnext = current_par_prov[:, 2].sum()
        Inext = len(G.nodes) - snext - rnext - dnext
        return current_par_prov, snext, rnext, dnext, Inext, connext, misnext

    def simulation_propagationOvertime(self,prev_par_prob,givenDebunkers,Timeinject,Timeinjectne):
        time_list = []
        time_list.append(0)

        snow = prev_par_prob[:, 0].sum()
        rnow = prev_par_prob[:, 1].sum()
        # connow = rev_par_prob[:, 4].sum()
        misnow = prev_par_prob[:, 5].sum()

        dnow = prev_par_prob[:, 2].sum()
        Inow = len(self.G.nodes) - snow - rnow - dnow

        current_listP = []  # [0]*tmax
        current_listP.append(prev_par_prob)

        reward6 = []  #
        reward6.append(Inow)  # int(r0))
        # what
        reward7 = []  #
        reward7.append(rnow)

        reward8 = []  #
        reward8.append(misnow)

        # for time_prev in range(1,tmax+1):
        for t in range(1, Timeinject):
            time_list.append(t)
            pre_prob = current_listP[t - 1]

            curr_prob, snext, rnext, dnext, \
            Inext, connext, misnext = self.simulation_propagation(pre_prob,givenDebunkers,Timeinjectne)

            reward6.append(Inext)
            reward7.append(rnext)
            reward8.append(misnext)
            current_listP.append(curr_prob)
        return curr_prob,snext, rnext,Inext, reward6,reward7,reward8, current_listP

    def close(self):
        pass


######### test
env = socialEnv()
# If the environment don't follow the interface, an error will be thrown
check_env(env, warn=True)


from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=250)

# Test the trained agent
# using the vecenv
obs,_ = env.reset()
n_steps = 20
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, terminated, truncated, info = env.step(action)
    #print("obs=", obs, "reward=", reward, "done=", done)
    done = terminated or truncated
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break

obs, _ = env.reset()

print(env.observation_space)
print(env.action_space)
print(env.action_space.sample())


n_steps = 10
for step in range(n_steps):
    print(f"Step {step + 1}")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print([obs.shape,reward*100])
    done = terminated or truncated
    #print("obs=", obs, "reward=", reward, "done=", done)
    if done:
        print("Goal reached!", "reward=", reward)
        break

