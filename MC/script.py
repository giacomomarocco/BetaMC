#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:49:33 2023

@author: giacomomarocco
"""

from betaDecay import BetaDecay
import numpy as np

q = np.array(11)
energy = np.array([5,2])
cth = np.array([0.6,0.1])
nucth = np.array([0.1,0.1])
nuphi = np.array([1,1])

BetaDecay(q, energy,cth,nucth,nuphi)

