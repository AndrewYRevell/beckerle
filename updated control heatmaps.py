# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import seaborn as sns

col = list(range(2,118))
correlationMatrix = np.zeros(shape = (12,12))
row = 0
ctrlList = []

for i in [x for x in range(4,17) if x != 10]:
    file = f"ctrl{i}.txt"
    data = np.loadtxt(file, skiprows=2, usecols = (col))
    data[np.where(data == 0)] = 1
    logged = np.log10(data)
    macs = np.amax(logged)
    heat = logged/macs
    #sns.heatmap(heat,square=True,cbar=False)#
    a = heat[np.triu_indices_from(heat,k=1)]
    column = 0
    for l in [x for x in range(4,17) if x != 10]:
        if l != i:
            file1 = f"ctrl{l}.txt"
            data1 = np.loadtxt(file1, skiprows=2, usecols = (col))
            data1[np.where(data1 == 0)] = 1
            logged1 = np.log10(data1)
            macs1 = np.amax(logged1)
            heat1 = logged1/macs1
            a1 = heat1[np.triu_indices_from(heat1,k=1)]
            
            aa = scipy.stats.spearmanr(a,a1)[0]
            correlationMatrix[row,column] = aa
        column = column + 1
    row = row + 1
    ctrlList.append(i)
sns.heatmap(correlationMatrix, square=True,xticklabels=(ctrlList),yticklabels=(ctrlList))
        
"""
saveimg / save plots?
"""
