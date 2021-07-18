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

#%% Parameters



seed = 0 #ROI number where activity starts
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
sns.displot(neighbor_sum_distribution[1,:] ,binwidth=0.05); plt.xlim(-1, 1)
sns.displot(neighbor_sum_distribution[2,:] ,binwidth=0.05); plt.xlim(-1, 1)
sns.displot(neighbor_sum_distribution[3,:] ,binwidth=0.05); plt.xlim(-1, 1)



"""
N # Total population, N.
I0 = 1 #Initial Infected
R0 = 0 #Initial Recovered
S0 = N-I0-R0 #Susceptibility
beta=np.mean(C) #probability of infection/contact rate
gamma=.2 #mean recovery rate
#%% 

# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Initial conditions vector
y0 = S0, I0, R0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
ax.set_xlabel('Time')
ax.set_ylabel('Number')
ax.set_ylim(0,200)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()
"""