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
T=[36,37,38,39,40,41,54,55,78,79,80,81,82,83,84,85,86,87,88,89]
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
"""        
        for L in range(90,95):
        if node_state[L, T[0]] == 1:
            print("HC, Left")
        if node_state[L, T[1]] == 1:
            print("HC, Right")
        if node_state[L, T[2]] == 1:
            print("PHC, Left")
        if node_state[L, T[3]] == 1:
            print("PHC, Right")
        if node_state[L, T[4]] == 1:
            print("Amygdala, Left")
        if node_state[L, T[5]] == 1:
            print("Amygdala, Right")
        if node_state[L, T[6]] == 1:
            print("FUSI, Left")
        if node_state[L, T[7]] == 1:
            print("FUSI, Right")
        if node_state[L, T[8]] == 1:
            print("HES, Left")
        if node_state[L, T[9]] == 1:
            print("HES, Right")
        if node_state[L, T[10]] == 1:
            print("T1, Left")
        if node_state[L, T[11]] == 1:
            print("T1, Right")
        if node_state[L, T[12]] == 1:
            print("T1P, Left")
        if node_state[L, T[13]] == 1:
            print("T1P, Right")
        if node_state[L, T[14]] == 1:
            print("T2, Left")
        if node_state[L, T[15]] == 1:
            print("T2, Right")
        if node_state[L, T[16]] == 1:
            print("T2P, Left")
        if node_state[L, T[17]] == 1:
            print("T2P, Right")
        if node_state[L, T[18]] == 1:
            print("T3, Left")
        if node_state[L, T[19]] == 1:
            print("T3, Right")    
"""
print(R)
print(I)
print(node_state[0,:])
print(node_state[1,:])
print(node_state[2,:])
print(node_state[3,:])
#sns.displot(neighbor_sum_distribution[1,:] ,binwidth=0.05); plt.xlim(-1, 1)
#sns.displot(neighbor_sum_distribution[2,:] ,binwidth=0.05); plt.xlim(-1, 1)
#sns.displot(neighbor_sum_distribution[3,:] ,binwidth=0.05); plt.xlim(-1, 1)



