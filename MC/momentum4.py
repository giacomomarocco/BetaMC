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
        self.threeMomentum = self.p[-3:]
        self.momentumMag = np.linalg.norm(self.threeMomentum)
        self.energy = self.p[0]
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
        modP = np.sqrt(e**2 - m**2)
        s_theta = np.sqrt(1 - c_theta**2)
        p = np.array([e, modP*s_theta*np.cos(phi),
            modP*s_theta*np.sin(phi), modP*c_theta])
        return cls(p, m)