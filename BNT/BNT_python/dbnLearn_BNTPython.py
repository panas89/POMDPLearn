#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 08:07:18 2017

@author: ppetousis
"""

import pandas as pd
import numpy as np

# import data
df = pd.read_csv('./dfDBNTrain.csv')

# each column  should correspond to a slice
# defining nodes of each slice in temporal order
cols =['Cancer', 'LDCT', 'InterventionObservation',
       'Cancer_1', 'LDCT_1', 'InterventionObservation_1',
       'Cancer_2', 'LDCT_2', 'InterventionObservation_2',
       'Cancer_3', 'LDCT_3', 'InterventionObservation_3', 
       'Cancer_4', 'LDCT_4', 'InterventionObservation_4']

df = df[cols]

# values start at zero in matlab start at 1, add 1
df = df + 1

# check number of null values
df.isnull().sum()

#optional, saving dataset
df.to_csv('./dfDBNTrainNum.csv',index=False,header=False )

# convert to numpy array
data = df.as_matrix()

# import octave to python package
from oct2py import octave

# define DBN model
intraLength = 3 # Length, in terms of nodes, of time=t slice
interLength = 3 # Length, in terms of nodes, of time=t-1 and time t slices
ns = [3,8,2]    # number of observations per node
horizon = 5     # number of time steps (i.e., number of slices)
max_iter = 100  # number of iterations

# octave script to create and learn DBN components
out = octave.dbnPOMDP_BNT(intraLength, interLength, ns, horizon, data, max_iter)

DBNcomponents = out.tolist()

priors = DBNcomponents[0][0]
O = np.concatenate((DBNcomponents[0][1],DBNcomponents[0][2]),axis=1)
T = DBNcomponents[0][3]
