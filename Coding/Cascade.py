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
row=0
column=0
nodesArray = []
#correlationMatrix = np.zeros(shape = (25,116))
for a in [x for x in range(4,17) if x != 10]:
    file = f"ctrl{a}.txt"
    
    C = np.loadtxt(file, skiprows=2, usecols = (col))
#Preprocess connectivity matrices
#log normalizing
    C[np.where(C == 0)] = 1
    C = np.log10(C)
    C = C/np.max(C)
#sns.heatmap(C)

#Thresholding
    threshold = 0.97 #bottom X percent of edge weights are eliminated
    C_thresh = copy.deepcopy(C)
    number_positive_edges = len(np.where(C > 0)[0])
    cutoff = int(np.round(number_positive_edges*threshold))
    positive_edges = C[np.where(C > 0)]
    cutoff_threshold = np.sort(positive_edges)[cutoff]
    C_thresh[np.where(C_thresh < cutoff_threshold)] = 0
    len(np.where(C_thresh > 0)[0])

#Cascading
#parameters
    seed = 36 #ROI number where activity starts
    time_steps = 25 #number of time steps before termination of simulation
    N = len(C_thresh) #number of nodes
    node_state = np.zeros(shape = (time_steps, N))
    node_state[0, seed] = 1 #make seed active
    cascading_denominator = np.max(C)/0.3 # converts edge weights into probabilities

    activation_probability_threshold_distribution = np.zeros(shape = (time_steps, N))
#%%
    for t in range(1, time_steps):
        #print(t)
        
        for i in range(N): #loop thru all nodes
            
            #find neighbors of node i
            previous_state = node_state[t-1,i]
            activation_probability = random.betavariate(1.2, 1.2) #random.uniform(0, 1) #probability of being activated
            neighbors = np.where(C_thresh[i,:] > 0)
            neighbors_weight = C_thresh[i,neighbors] 
            neighbors_state = node_state[t-1, neighbors]
            neighbors_probability = neighbors_weight/cascading_denominator #convets edge weights into probabilites
            activation_probability_threshold = 1-np.prod((1-(neighbors_probability * neighbors_state))) #the proabbility that any nieghbors activates the node
        
            if activation_probability <= activation_probability_threshold: #if sum is greater than threshold, make that node active
                node_state[t, i] = 1
            if activation_probability > activation_probability_threshold:
                node_state[t, i] = 0
            if previous_state == 1:
                node_state[t, i] = 1
        activation_probability_threshold_distribution[t,i] = activation_probability_threshold
    b=node_state[:,:]
    d=node_state.flatten() #time_steps x nodes (116)?
    
    
    nodesArray.append(d)
    row=row+1
    #print(node_state[24,:])
    P = np.count_nonzero(node_state[24,:])
    print(P/116)

#print(nodesArray)
f=np.array(nodesArray)
print(f)
aa = scipy.stats.spearmanr(f)
print(aa)
print(len(nodesArray))




