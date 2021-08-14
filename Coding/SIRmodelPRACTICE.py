# -*- coding: utf-8 -*-

import os
from scipy.io import loadmat
import numpy as np
import copy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from scipy import stats
from scipy.integrate import odeint

col = list(range(2,118))
C = np.loadtxt("ctrl16.txt", skiprows=2,usecols=col)
#%% Preprocess connectivity matrices
#log normalizing
C[np.where(C == 0)] = 1
C = np.log10(C)
C = C/np.max(C)
N = len(C) #number of nodes
#sns.heatmap(C,square=True)

#thresholding
threshold = .4 #bottom X percent of edge weights are eliminated
C_thresh = copy.deepcopy(C)
number_positive_edges = len(np.where(C > 0)[0])
cutoff = int(np.round(number_positive_edges*threshold))
positive_edges = C[np.where(C > 0)]
cutoff_threshold = np.sort(positive_edges)[cutoff]
C_thresh[np.where(C_thresh < cutoff_threshold)] = 0
len(np.where(C_thresh > 0)[0])

#%% Parameters
threshold=.02
seed = 36 #ROI number where activity starts
time_steps = 25 #number of time steps before termination of simulation
node_state = np.zeros(shape = (time_steps, N))
node_state[0, seed] = 1 #make seed active


I=1
R=0


neighbor_sum_distribution = np.zeros(shape = (time_steps, N))
for t in range(1, time_steps):
    
    for i in range(N):
        beta=random.betavariate(1,np.mean(C))#better determinant of beta?
        if node_state[t,i] >= 0:
            previous_state = node_state[t-1,i]
            neighbors = np.where(C[i,:] > 0)
            neighbors_weight = C[i,neighbors] 
            neighbors_state = node_state[t-1, neighbors]
            neighbors_sum = np.sum(neighbors_weight * neighbors_state)
            strength = np.sum(neighbors_weight)
            if neighbors_sum >= beta: #if sum is greater than threshold, make that node active
                node_state[t, i] = 1
            if neighbors_sum < beta:
                node_state[t, i] = 0
            if previous_state == 1:
                node_state[t, i] = -1
            if previous_state==-1:
                node_state[t,i]=-2
            
        
            
        neighbor_sum_distribution[t,i] = neighbors_sum/strength
        
        
print(R)
print(I)
print(node_state[0,:])
print(node_state[1,:])
print(node_state[2,:])
print(node_state[3,:])
#sns.displot(neighbor_sum_distribution[1,:] ,binwidth=0.05); plt.xlim(-1, 1)
#sns.displot(neighbor_sum_distribution[2,:] ,binwidth=0.05); plt.xlim(-1, 1)
#sns.displot(neighbor_sum_distribution[3,:] ,binwidth=0.05); plt.xlim(-1, 1)



