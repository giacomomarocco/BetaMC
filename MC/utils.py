#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:22:32 2023

@author: giacomomarocco
"""
import numpy as np


def direction_from_polar(c_theta,phi):
    s_theta = np.sqrt(1-c_theta**2)
    pHat = np.array([s_theta*np.cos(phi),s_theta*np.sin(phi),c_theta])
    return pHat

def momentumMagnitude(energy, mass):
    return np.sqrt(energy**2-mass**2)

def generate_samples(*x, n_samples):
    dimension = len(x)
    samples = np.empty([n_samples,dimension])
    for sampleIndex in range(n_samples):
        for variableIndex in range(dimension):
            samples[sampleIndex,variableIndex] = np.random.choice(x[variableIndex])
    return samples