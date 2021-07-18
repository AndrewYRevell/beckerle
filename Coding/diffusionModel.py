# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 18:21:07 2021
@author: arevell
"""

import os
from scipy.io import loadmat
import numpy as np
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

col = list(range(2,118))
C = np.loadtxt("ctrl4.txt", skiprows=2,usecols=col)
#%% Preprocess connectivity matrices
#log normalizing
C[np.where(C == 0)] = 1
C = np.log10(C)
C = C/np.max(C)
#sns.heatmap(C)
#%%
#Thresholding
threshold = 0.5 #bottom X percent of edge weights are eliminated
C_thresh = copy.deepcopy(C)
number_positive_edges = len(np.where(C > 0)[0])
cutoff = int(np.round(number_positive_edges*threshold))
positive_edges = C[np.where(C > 0)]
cutoff_threshold = np.sort(positive_edges)[cutoff]
C_thresh[np.where(C_thresh < cutoff_threshold)] = 0
len(np.where(C_thresh > 0)[0])
#%%
#parameters
seed = 0 #ROI number where activity starts (value of 0-115)
time_steps = 5 #number of time steps before termination of simulation
N = len(C_thresh) #number of nodes
node_state = np.zeros(shape = (time_steps, N))
node_state[0, seed] = 1 #make seed active
#%%
#LTM
threshold_LTM = .9 #Threshold in linear threshold model - the cutoff where a node can become in active state
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
        strength = np.sum(neighbors_weight)
        if neighbors_sum >= threshold_LTM: #if sum is greater than threshold, make that node active
            node_state[t, i] = 1
        if neighbors_sum < threshold_LTM:
            node_state[t, i] = 0
        if previous_state == 1:
            node_state[t, i] = 1
        neighbor_sum_distribution[t,i] = neighbors_sum/strength 

sns.displot(neighbor_sum_distribution[0,:] ,binwidth=0.05); plt.xlim(0, 1)
sns.displot(neighbor_sum_distribution[1,:] ,binwidth=0.05); plt.xlim(0, 1)
sns.displot(neighbor_sum_distribution[2,:] ,binwidth=0.05); plt.xlim(0, 1)
sns.displot(neighbor_sum_distribution[3,:] ,binwidth=0.05); plt.xlim(0, 1.2)
print(node_state[0,:])
print(node_state[1,:])
print(node_state[2,:])
print(node_state[3,:])
#%%
"""
#Cascading
#parameters
seed = 36 #ROI number where activity starts
time_steps = 25 #number of time steps before termination of simulation
N = len(C_thresh) #number of nodes
node_state = np.zeros(shape = (time_steps, N))
node_state[0, seed] = 1 #make seed active
#%%
cascading_denominator = np.max(C)/0.3 # converts edge weights into probabilities
#%%
activation_probability_threshold_distribution = np.zeros(shape = (time_steps, N))
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
 
#node_state_df = pd.DataFrame(node_state, columns = col['label'])
#node_state_df.to_csv(os.path.join (path, "blender", "diffusion_models_simulation", "sub-RID0278_AAL_seed36.csv"), index=False)
Nnumbers = 10000
numbers = np.zeros(shape = Nnumbers)
for k in range(Nnumbers):
    numbers[k] = random.betavariate(2,2)
np.mean(numbers)
sns.histplot(numbers)
"""