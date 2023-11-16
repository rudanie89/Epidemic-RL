import numpy as np
import random
import networkx as nx
from networkx.generators.community import LFR_benchmark_graph
import time
import math

import gym
#from gym import Env
from gym.spaces import Discrete, Box, MultiDiscrete
import numpy as np
import random
import os

from collections import Counter
from itertools import count
import json

#https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.LFR_benchmark_graph.html
# G = nx.generators.random_graphs.binomial_graph(100,0.02,seed=246,directed=False) #seed=246,
G = nx.generators.random_graphs.binomial_graph(100,0.02,seed=240,directed=False) #seed=246,
#LFR_benchmark_graph(100, 3, 1.5, 0.5, average_degree=5.5, min_community=20, seed=246)
len(list(G.edges())) #1104
nx.is_directed(G) # False


neigDict = {}
for i in G.nodes:
    neigbor1 = list(G.neighbors(i))
    neigDict[i] =len(neigbor1)

edgeAll = list(G.edges())
len(edgeAll)

IRnodes = list(G.nodes)
len(IRnodes)

nodeJa = dict()

start= time.time()
for i in G.nodes:
    nodeI = dict()
    if len(list(G.neighbors(i))) != 0:
        nodeI = [round(list(nx.jaccard_coefficient(G, [(i, p)]))[0][2], 5) for p in list(G.neighbors(i))]
        nodeJa[i] = dict(zip(list(G.neighbors(i)), nodeI))

end = time.time()
print("time to complete!")
print(end - start)

############### initial values for user belief ################
#
# beliefMis = dict()
# beliefDe = dict()
#
# # # 10% of nodes: let's choose 524 first nodes that have high belief of misinformation with prob in [0.7, 1]
# first10p = random.sample(IRnodes, 10)
# for i in first10p:
#     beliefMis[i] = round(random.uniform(0.7, 1.0), 3)
#     beliefDe[i] = round(1 - beliefMis[i], 3)
#
# len(first10p)
# # # next 10% of nodes: let's choose next 524 nodes (from ) that have high belief of counter information
# IRnodes2 = [u for u in IRnodes if u not in first10p]
# len(IRnodes2)
# #
# next10p = random.sample(IRnodes2, 10)
# for i in next10p:
#     beliefDe[i] = round(random.uniform(0.7, 1.0), 3)
#     beliefMis[i] = round(1 - beliefDe[i], 3)
#
# IRnodes3 = [u for u in IRnodes2 if u not in next10p]
# len(IRnodes3)
# #
# for i in IRnodes3:
#     beliefMis[i] = 0
#     beliefDe[i] = 0
#
# # ## save all initial information to file
# first10pdict ={'first10p': first10p}
#
# pathfirst10p = "D:\\project4\\codefirsttry\\synthetic\\outputjunebi\\net100\\belief\\first10p.txt"  # debunker10_14au
#
# with open(pathfirst10p, 'w+') as ffirst10p:
#     json.dump(first10pdict, ffirst10p, indent=4)
#
# next10pdict ={'next10p': next10p}
# pathnext10p = "D:\\project4\\codefirsttry\\synthetic\\outputjunebi\\net100\\belief\\next10p.txt"  # debunker10_14au
#
# with open(pathnext10p, 'w+') as fnext10p:
#     json.dump(next10pdict, fnext10p, indent=4)

# Save bias belief
pathbeliefMis = "D:\\project4\\codefirsttry\\synthetic\\outputjunebi\\net100\\belief\\beliefMis.txt"  # debunker10_14au
#
# with open(pathbeliefMis, 'w+') as fbeliefMis:
#     json.dump(beliefMis, fbeliefMis, indent=4)

pathbeliefDe = "D:\\project4\\codefirsttry\\synthetic\\outputjunebi\\net100\\belief\\beliefDe.txt"  # debunker10_14au
## this value will not exist anymore because of my mistake to save them withou beliefDe
#
# with open(pathbeliefDe, 'w+') as fbeliefDe:
#     json.dump(beliefDe, fbeliefDe, indent=4)


##loading bias belief
with open(pathbeliefDe, 'r') as fbeliefDe:
    databeliefDe = fbeliefDe.read()
beliefDe1 = json.loads(databeliefDe)

#pathbeliefMis = "E:\\rl\\dqn1\\irvine\\mist\\output\\beliefMis.txt"
with open(pathbeliefMis, 'r') as fbeliefMis:
    databeliefMis = fbeliefMis.read()
beliefMis1 = json.loads(databeliefMis)

beliefDe=dict(zip([int(i) for i in list(beliefDe1.keys())], list(beliefDe1.values())))
beliefMis=dict(zip([int(i) for i in list(beliefMis1.keys())], list(beliefMis1.values())))
nodeBmis = [i for i in list(beliefMis.keys()) if beliefMis[i]==0]
len(nodeBmis) #


############# incoming nodes
indegreeDict = {}
proindegreeDict={}
for i in G.nodes:
    indegreeDict[str(i)] = [tup[0] for tup in edgeAll if tup[1] == i]
    proindegreeDict[str(i)] = 0.1 #round(1/(len(indegreeDict[str(i)])+0.001),6)

################# algorithm 1 - update belief from MIST paper ###########
def update_B(G, Mis, De, alpha, R, biasMis, biasDe):  # cu: (G,Mis,De,alpha,degree,R):
    userState_At = dict()
    RM_later = R
    DM_all = Mis + De

    for i in G.nodes:
        if i in Mis:
            userState_At[i] = -1
        elif i in De:
            userState_At[i] = 1
        else:
            userState_At[i] = 0

    Inode = 0
    Rnew = []
    for t in range(1, alpha):
        Delater = []
        Mislater = []
        #userState_t = dict()
        userbeliefD_t = dict()
        userbeliefM_t = dict()

        for v in R: ## actually, R also need to be updated by time t
            A = [u for u in indegreeDict[str(v)] if ((len(indegreeDict[str(v)])!=0) and (userState_At[u]==1))] #(u in De))]
            if len(A) == 0:  # A is None:
                userbeliefD_t[v] = 0
            else:
                userbeliefD_t[v] = 1 - np.prod([(1 - proindegreeDict[str(v)] * biasDe[v]) for u_i in A])

            F = [u for u in indegreeDict[str(v)] if ((len(indegreeDict[str(v)])!=0) and (userState_At[u]==-1))] #(u in Mis))]
            if len(F) == 0:  # A is None:
                userbeliefM_t[v] = 0
            else:
                userbeliefM_t[v] = 1 - np.prod([(1 - proindegreeDict[str(v)] * biasMis[v]) for u_i in F])
        for v in R:
            if userbeliefD_t[v] >= userbeliefM_t[v] + 0.0001: #0.01: #0.02:
                userState_At[v] = 1
                Delater.append(v)
            elif userbeliefM_t[v] > userbeliefD_t[v] + 0.0001: #0.01: #0.02:
                Mislater.append(v)
                userState_At[v] = -1
            else:
                userState_At[v] = 0
                Rnew.append(v)
        DM_all = Delater + Mislater
        RM_later = [u for u in IRnodes if u not in DM_all]

        if t < alpha:
            Inode = Mislater
    return userState_At, Delater, Mislater, DM_all, RM_later, Inode

#### calculating bias for each user ############
#### correct way of calculating the biasRegular
def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

reNode = [i for i in G.nodes if ((beliefDe[i] == 0) and (beliefMis[i] == 0))]

def biasRegular(G, reNode, nodeJa, biasInD, biasInM):

    biasDe_list = []
    biasDe_list.append(biasInD)

    biasMis_list = []
    biasMis_list.append(biasInM)

    for t in range(1, 7):
        prevD = biasDe_list[t - 1]
        prevM = biasMis_list[t - 1]

        nextDe1 = [biasInD[i] for i in list(G.nodes) if i not in reNode]
        uother = [i for i in list(G.nodes) if i not in reNode]
        next_biasDe1 = dict(zip(uother, nextDe1))

        nextDe2 = [biasInD[i] if (len(indegreeDict[str(i)]) == 0) else (np.sum(
            [prevD[p] * nodeJa[i][p] for p in indegreeDict[str(i)]])) / (len(indegreeDict[str(i)]) + 0.001) for i in
                   reNode]
        next_biasDe2 = dict(zip(reNode, nextDe2))

        next_biasDe = Merge(next_biasDe1, next_biasDe2)

        nextM1 = [biasInM[i] for i in list(G.nodes) if i not in reNode]
        next_biasM1 = dict(zip(uother, nextM1))

        nextM2 = [biasInM[i] if (len(indegreeDict[str(i)]) == 0) else (np.sum(
            [prevM[p] * nodeJa[i][p] for p in indegreeDict[str(i)]])) / (len(indegreeDict[str(i)]) + 0.001) for i in
                  reNode]
        next_biasM2 = dict(zip(reNode, nextM2))
        next_biasM = Merge(next_biasM1, next_biasM2)
        biasDe_list.append(next_biasDe)
        biasMis_list.append(next_biasM)
        # else:

    return biasDe_list[-1], biasMis_list[-1]

biasDe, biasMis = biasRegular(G, reNode, nodeJa, beliefDe, beliefMis)

################ let's creating env for mist approach ################
#################################################################
neigDict = {}
for i in G.nodes:
    neigbor1 = list(G.neighbors(i))
    neigDict[i] =len(neigbor1)

top50list=[]
for k in dict(Counter(neigDict).most_common(50)).keys():
    top50list.append(k)

edgeAll = list(G.edges())
len(edgeAll)

IRnodes = list(G.nodes)
len(IRnodes)

def areSame(A,B):
    C=np.abs(A-B)
    d=C<=epsilon
    count_d=len(C[d])
    if count_d >= 290: #let's relax this to make sure not picking a lot of do nothing; before was: 24230
        return 0 #
    return 1
#nodes: 5242; edges:

##outdegree############
outdegreeDict = {}
for i in G.nodes: #G.nodes:
    neigbor2 = [tup[1] for tup in edgeAll if tup[0] == i]
    outdegreeDict[i] =len(neigbor2)
'''
Betweenness 
'''
# betCent = nx.betweenness_centrality(G, normalized=True, endpoints=True)
# print(type(betCent))
# betCent[107]
#
# top20list=[] # top10 from whole network have highest betweenness: [8, 399, 104, 31, 102, 41, 2, 40, 522, 248]
# for k in dict(Counter(betCent).most_common(20)).keys():
#     top20list.append(k)

'''
Creating Env
'''
class socialEnv(gym.Env): #object):
    def __init__(self):
        self.G=G
        self.biasMis=biasMis
        self.biasDe=biasDe
        self.edgeAll=edgeAll
        self.infectednode = None #random.sample(initial_infected, 2)

        self.debunkersU = []
        # self.mitigators = top20list

        self.muR = None #etaR
        self.muT = None #etaT
        self.action_space = Discrete(len(list(G.nodes))) #Discrete(788)
        self.current_state = None
        self.debunkers = None
        self.R = None

        self.Sinfected = None

        self.budgetAll=20
        self.dem_inf = 0
        self.S_u = None

        self.Sinfected = None
        self.illegalnodes = None
        self.mitigators = None
        self.infectednodesNei = []
        self.listneib = []

    def reset(self): #,iepisode
        # self.mitigators = top20list  # top50list[0:10] #list(self.G.nodes) #top50list[0:20]
        self.current_state,_,self.illegalnodes, self.mitigators,_ = self.initialstatefunc() #iepisode) #
        self.budgetAll = 20
        #edgeAll = list(self.G.edges())

        return self.current_state #, len(edgeAll)

    def initialstatefunc(self): #,mitigation #,iepisode
        #self.mitigators = top20list  # top50list[0:10] #list(self.G.nodes) #top50list[0:20]
        #self.infectednode = random.sample([ui for ui in list(self.G.nodes) if ui not in mitigation], 5)
        self.Sinfected = []
        while len(self.Sinfected) < 5:
            self.infectednode = random.sample([ui for ui in list(self.G.nodes)], 10) #[657,3171,1870,690,2339,815,120,3305,2825,3715]
            #random.sample([ui for ui in list(self.G.nodes)], 10)
            self.R = [user for user in list(G.nodes) if user not in self.infectednode]  # (user not in self.mitigators) and

            userState_AtS, DelaterS, MislaterS, DM_allS, RM_laterS, IsetS = update_B(self.G, self.infectednode, [], 10,
                                                                                     self.R, self.biasMis, self.biasDe)

            userState0_A = {}
            # for i in self.G:
            #     userState0_A[i] = self.current_state[i][0]
            for i, state in userState_AtS.items():
                userState0_A[i] = state

            self.ini_par_prob = np.zeros((len(self.G.nodes), 1), dtype=np.float32)
            for i in self.G.nodes:
                self.ini_par_prob[i][0] = userState0_A[i]

            self.Sinfected = [v for v in self.R if userState0_A[v] == -1]
            #print("len of infected nodes:", len(self.Sinfected))

            if len(self.Sinfected)>=5:
                self.current_state = self.ini_par_prob
                initialInfected = list(self.Sinfected+self.infectednode)
                # self.mitigators = [i for i in list(self.G.nodes) if i not in initialInfected]
                self.mitigators = random.sample([i for i in list(self.G.nodes) if i not in initialInfected],50)
                self.illegalnodes = [inode for inode in list(G.nodes) if inode not in self.mitigators]
                #self.infectednodes_epi
                self.listneib = []
                for i in self.Sinfected:
                    self.listneib = self.listneib + list(nx.all_neighbors(G, i))

                self.infectednodesNei = [i for i in self.listneib if i not in self.infectednode]
                print("Length of neighbor of infected nodes", len(list(set(self.infectednodesNei))))

                break

        return self.current_state, len(self.Sinfected),self.illegalnodes, self.mitigators,list(set(self.infectednodesNei))

    def infectedreset(self):
        return self.Sinfected,self.infectednode,self.mitigators,list(set(self.infectednodesNei))

    def step(self, action):
        # self.current_state = self.current_state
        # self.Sinfected = self.Sinfected
        # ud = action
        #
        # if action not in self.mitigators:
        #     self.S_u=[]
        #     self.budgetAll = self.budgetAll
        #     RLreward = -1
        #     done = False
        # else:
        #legal_action = self.check_legal(action)
        nodeSaved = dict()
        IsetD = dict()

        recNodesaved = dict()
        recoveredNode = dict()
        ud = action #legal_action #self.mitigators[action]
        current_Reg = [u for u in self.R if u != ud]  # action
        # ud = action #self.mitigators[action]
        userState_At1, Delater1, Mislater1, DM_all1, RM_later1, Iset = update_B(self.G, self.infectednode, [ud], 25,
                                                                                current_Reg, self.biasMis, self.biasDe)

        IsetD[action] = Iset
        userState_de = {}  # userState0
        # for i in self.G:
        #     userState_de[i] = self.current_state[i][0]

        for i, state in userState_At1.items():
            userState_de[i] = state

        current_Sinfected = self.Sinfected

        self.S_u = [v for v in current_Reg if ((v in current_Sinfected) and (userState_de[v] == 1))]
        Srecoved = [v for v in current_Reg if userState_de[v] == 1]

        nodeSaved[action] = len(self.S_u)
        recNodesaved[ud] = self.S_u

        recoveredNode[ud] = len(Srecoved)

        self.budgetAll = self.budgetAll - 1

        done = bool(self.budgetAll <= 0)

        if not done:
            RLreward = round((len(self.S_u) / (len(self.Sinfected) + 0.001)), 4)
        else:
            RLreward = round((len(self.S_u) / (len(self.Sinfected) + 0.001)), 4)

        ##update for the current state
        self.current_state = np.zeros((len(self.G.nodes), 1), dtype=np.float32)
        for i in self.G.nodes:
            self.current_state[i][0] = userState_de[i]

        self.Sinfected = [u_k for u_k in current_Sinfected if u_k not in self.S_u]
        ## update current regular nodes
        # self.R = list(set(self.R) - set(S_u))

        return self.current_state, round(RLreward, 4), done, self.budgetAll, self.infectednode, len(self.S_u), len(self.Sinfected),len(current_Sinfected), ud, {}

# env = socialEnv()
# last_observation = env.reset()