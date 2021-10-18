# -*- coding: utf-8 -*-

import os
from scipy.io import loadmat
import numpy as np
import copy
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import random

col = list(range(2,118))
choiceArray=["L","R"]
nodesArray=[]
ctrlList=[]

#RIDList=[37,46,89,139,194,213,278,309,320,341,365,380,420,440,454,459,490,502,508,520,522,529,536,572,583,595,596,648]
RIDList=[]

#%% Controls
for a in [x for x in range(4,17) if x != 10]:
    side=np.random.choice(choiceArray)
    print(side)
    if side=="R":
       seed=37 #ROI number where activity starts
    if side=="L":
        seed=36
    ctrlList.append(a)
    file = f"ctrl{a}.txt"
    C = np.loadtxt(file, skiprows=2, usecols = (col))
#Preprocess connectivity matrices
#log normalizing
    C[np.where(C == 0)] = 1
    C = np.log10(C)
    C = C/np.max(C)
#sns.heatmap(C)

#Thresholding
    threshold = .2 #bottom X percent of edge weights are eliminated
    C_thresh = copy.deepcopy(C)
    number_positive_edges = len(np.where(C > 0)[0])
    cutoff = int(np.round(number_positive_edges*threshold))
    positive_edges = C[np.where(C > 0)]
    cutoff_threshold = np.sort(positive_edges)[cutoff]
    C_thresh[np.where(C_thresh < cutoff_threshold)] = 0
    len(np.where(C_thresh > 0)[0])


#parameters
    g=.001
    #seed=36
    time_steps = 200 #number of time steps before termination of simulation
    N = len(C_thresh) #number of nodes
    node_state = np.zeros(shape = (time_steps, N))
    node_state[0, seed] = 1 #make seed active
    neighbor_sum_distribution = np.zeros(shape = (time_steps, N))

    for t in range(1, time_steps):
        #print(t)
        for i in range(N): #loop thru all nodes
            #find neighbors of node i
            previous_state = node_state[t-1,i]
            neighbors = np.where(C_thresh[i,:] > 0)
            neighbors_weight = C_thresh[i,neighbors] 
            neighbors_state = node_state[t-1, neighbors]
            neighbors_sum = np.sum(neighbors_weight * neighbors_state)
            node_state[t,i] = previous_state + (neighbors_sum * g)
            if node_state[t,i] >= 1:
                node_state[t,i] = 1
            
    b=node_state[:,:]
    d=node_state.flatten() #time_steps x nodes (116)?
    nodesArray.append(d)
    #print(node_state[time_steps-1,:])
    #print(np.all(node_state[time_steps-1,:]==1)) #check if all nodes are fully active


#%%  Patients
MTLList=[89,320,341,341,365,490,508,522,595,596] 
NEList=[139,194,309,440,459,508,520,529] #508 is both 
RIDList.extend(MTLList)
RIDList.extend(NEList)
RIDRIGHT = [194,309,341,522,529] #341 bilateral

for a in RIDList:
    file = f"RID{a}.txt"
    ctrlList.append(a)
    C = np.loadtxt(file, skiprows=2, usecols = (col))
#log normalizing
    C[np.where(C == 0)] = 1
    C = np.log10(C)
    C = C/np.max(C)
    #sns.heatmap(C)

#Thresholding
    threshold = 0.2 #bottom X percent of edge weights are eliminated
    C_thresh = copy.deepcopy(C)
    number_positive_edges = len(np.where(C > 0)[0])
    cutoff = int(np.round(number_positive_edges*threshold))
    positive_edges = C[np.where(C > 0)]
    cutoff_threshold = np.sort(positive_edges)[cutoff]
    C_thresh[np.where(C_thresh < cutoff_threshold)] = 0
    len(np.where(C_thresh > 0)[0])

#parameters
    if a in RIDRIGHT:
        seed=37
        if a==341: #341R comes first
            RIDRIGHT.remove(341)
    else:
        seed=36
    g=.001
    time_steps = 200 #number of time steps before termination of simulation
    N = len(C_thresh) #number of nodes
    node_state = np.zeros(shape = (time_steps, N))
    node_state[0, seed] = 1 #make seed active
    neighbor_sum_distribution = np.zeros(shape = (time_steps, N))

    for t in range(1, time_steps):
        #print(t)
        for i in range(N): #loop thru all nodes
            #find neighbors of node i
           previous_state = node_state[t-1,i]
           neighbors = np.where(C_thresh[i,:] > 0)
           neighbors_weight = C_thresh[i,neighbors] 
           neighbors_state = node_state[t-1, neighbors]
           neighbors_sum = np.sum(neighbors_weight * neighbors_state)
           node_state[t,i] = previous_state + (neighbors_sum * g)
           if node_state[t,i] >= 1:
                node_state[t,i] = 1
    b=node_state[:,:]
    d=node_state.flatten() #time_steps x nodes (116)?
    nodesArray.append(d)
    #print(node_state[24,:])
    #print(node_state[time_steps-1,:])
    #print(nodesArray)

#%% Correlation Matrixing
n = len(nodesArray)
correlationArray=np.zeros((n,n))
for x in range(n):
    for y in range(n):
        if y!= x:
            correlationArray[x,y]=scipy.stats.spearmanr(nodesArray[x],nodesArray[y])[0]

sns.heatmap(correlationArray, square=True,xticklabels=(ctrlList),yticklabels=(ctrlList))
a2 = correlationArray[np.triu_indices_from(correlationArray,k=1)]
print(np.median(a2))
print(np.std(a2))
