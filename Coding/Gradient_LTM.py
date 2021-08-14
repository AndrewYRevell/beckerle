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
nodesArray=[]
ctrlList=[]
for a in [x for x in range(4,17) if x != 10]:
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
    threshold = .4 #bottom X percent of edge weights are eliminated
    C_thresh = copy.deepcopy(C)
    number_positive_edges = len(np.where(C > 0)[0])
    cutoff = int(np.round(number_positive_edges*threshold))
    positive_edges = C[np.where(C > 0)]
    cutoff_threshold = np.sort(positive_edges)[cutoff]
    C_thresh[np.where(C_thresh < cutoff_threshold)] = 0
    len(np.where(C_thresh > 0)[0])

#Cascading
#parameters
    threshold=.02
    seed = 36 #ROI number where activity starts
    time_steps = 25 #number of time steps before termination of simulation
    N = len(C_thresh) #number of nodes
    node_state = np.zeros(shape = (time_steps, N))
    node_state[0, seed] = 1 #make seed active
    neighbor_sum_distribution = np.zeros(shape = (time_steps, N))
    gradient=0

    for t in range(1, time_steps):
        
        for i in range(N): #loop thru all nodes
            #find neighbors of node i
            previous_state = node_state[t-1,i]
            neighbors = np.where(C_thresh[i,:] > 0)
            neighbors_weight = C_thresh[i,neighbors] 
            neighbors_state = node_state[t-1, neighbors]
            neighbors_sum = np.sum((neighbors_weight * neighbors_state) + gradient)
            strength = np.sum(neighbors_weight)
            g = np.count_nonzero(node_state[t,:])
            #print(g)
            gradient=gradient+.1
            if neighbors_sum >= threshold*strength: #if sum is greater than threshold, make that node active
                node_state[t, i] = 1
            if neighbors_sum+gradient < threshold*strength:
                node_state[t, i] = 0
            if previous_state == 1:
                    node_state[t, i] = 1   
            #neighbor_sum_distribution[t,i] = neighbors_sum/strength
    b=node_state[:,:]
    d=node_state.flatten() #time_steps x nodes (116)?
    nodesArray.append(d)
    #print(node_state[24,:])
    P = np.count_nonzero(node_state[24,:])
    print(P/116)

n = len(nodesArray)
correlationArray=np.zeros((n,n))
for x in range(n):
    for y in range(n):
        if y!= x:
            correlationArray[x,y]=scipy.stats.spearmanr(nodesArray[x],nodesArray[y])[0]

sns.heatmap(correlationArray, square=True,xticklabels=(ctrlList),yticklabels=(ctrlList))
print(np.median(correlationArray))
print(np.std(correlationArray))
