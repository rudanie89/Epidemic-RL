#creating social network environment using gymnasium
import numpy as np
import math
import networkx as nx

import gymnasium as gym
from gymnasium import spaces

import random
import os
import json
from collections import Counter

class socialEnv(gym.Env):
    def __init__(self):
        super(socialEnv, self).__init__()
        self.G=nx.generators.random_graphs.binomial_graph(100,0.02,seed=246,directed=False) #xem co dung ko -- minh se thu chay lai cai nay 
        self.edgeAll=list(self.G.edges())
        self.infectednode = None
        self.debunkersU = None
        self.mitigators = top30list #cho nay phai sua lai het 

        self.muR = None #etaR
        self.muT = None #etaT
        self.action_space = spaces.Discrete(30)
        self.observation_space = spaces.Box(low=np.zeros((len(self.G.nodes), 6)).reshape(-1), high=np.ones((len(self.G.nodes), 6)).reshape(-1), shape=(600,), dtype=np.float64)
        # self.current_state = None
        self.debunkers = None

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
            self.ini_par_prob[i] = prob_ag

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

        return self.current_state.reshape(-1),{}

    def step(self, action):
        self.debunkers = self.mitigators[action]

        current_par_provM, snext, rnext, Inext, \
        reward6, reward7, reward8, current_listP = self.simulation_propagationOvertime(self.current_state, [self.debunkers],25,'False')
        self.budgetAll = self.budgetAll - 1*costu[self.debunkers] #*probdictT[self.debunkers]
        terminated = bool(self.budgetAll<=0)
        truncated = False
        RLreward = round((rnext/100),4)  
        self.current_state = current_par_provM
        self.current_I = Inext
        self.current_R = rnext

        return (self.current_state.reshape(-1), round(RLreward,4), terminated,truncated,{},)
      
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

                    if Timeinjectne=='True':
                        r = 0
                        v = (1 - negAll)
                    else:
                        v = (1 - negAll) * posAll
                        r = 1 - posAll
                    probSus_curr = min(1,round((1 - v - r) * prob_agPre[0],4))

                    probRec_curr = min(1,round(r * (1 - prob_agPre[1]) + prob_agPre[1], 4))
                    probDef_curr = min(1,round((1 - r) * prob_agPre[2], 4))  # self.prob_ag['def']
                    probCon_curr = min(1,round(prob_agPre[0] * v,4))  # max(prob_agPre[0] * v,1- (probSus_curr+probRec_curr)) #them vao
                    probAct_curr = min(1,round((1 - prob_agPre[1]) * r, 4))

                    probMis_curr = min(1,round((prob_agPre[4] + prob_agPre[5]) * (1 - r), 4))

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
        misnow = prev_par_prob[:, 5].sum()

        dnow = prev_par_prob[:, 2].sum()
        Inow = len(self.G.nodes) - snow - rnow - dnow

        current_listP = []  # [0]*tmax
        current_listP.append(prev_par_prob)

        reward6 = []  #
        reward6.append(Inow)
        reward7 = []  #
        reward7.append(rnow)

        reward8 = []
        reward8.append(misnow)
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
