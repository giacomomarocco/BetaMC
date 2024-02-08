#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:31:05 2023

@author: giacomomarocco
"""

import numpy as np

class Momentum4:
    def __init__(self, p, m):
        self.p = p
        self.m = m

    @classmethod
    def from_cartesian(cls, e, px, py, pz, m):
        p = np.array([e, px, py, pz])
        return cls(p, m)
    
    @classmethod
    def from_threeMomentum(cls,pVector,m):
        p = np.array([np.sqrt(m**2+np.dot(pVector, pVector)),pVector[0],pVector[1],pVector[2]])
        return cls(p,m)
        
    @classmethod
    def from_polar(cls, e, c_theta, phi, m):
        p = np.array([e, np.sqrt(e**2 - m**2)*np.sqrt(1 - c_theta**2)*np.cos(phi),
            np.sqrt(e**2 - m**2)*np.sqrt(1 - c_theta**2)*np.sin(phi), np.sqrt(e**2 - m**2)*c_theta])
        return cls(p, m)