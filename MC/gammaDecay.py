#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:31:12 2024

@author: giacomomarocco
"""

#This code contains the GammDecay class, a subclass of BetaDecay. 
#This means we do a usual beta decay, and then give the sphere some random momentum kicks due to the emitted photons. 
#Currently, this only keeps the recoil information, and ignores the photons. 

import numpy as np
from constants import electronMass
from constants import protonMass
from utils import direction_from_polar
from utils import momentumMagnitude
from momentum4 import Momentum4
from JTWParameters import JTW_Coefficients
from betaDecay import BetaDecay

class GammaDecay(BetaDecay):
    def __init__(self, gammaEnergy, QValue, electronEnergy, electronCosTheta, neutrinoCosTheta, neutrinoPhi, unpolarised = True):
        #Initialise the beta decay, but now the energy available for beta decay is less (QValue-gammaEnergy).
        self.gammaEnergy = gammaEnergy
        self.gammaMomentum()
        super().__init__(QValue-gammaEnergy, electronEnergy, electronCosTheta, neutrinoCosTheta, neutrinoPhi, unpolarised = True)
        
    def gammaMomentum(self):
        cosThetaGamma = np.random.uniform(-1.0,1.0)
        phiGamma = np.random.uniform()*2*np.pi
        self.gammaMomentum = Momentum4.from_polar(self.gammaEnergy, cosThetaGamma, phiGamma, 0).threeMomentum
        return self
    
    def childMomentum(self):
        self.childMomentum = -(self.neutrinoMomentum + self.gammaMomentum + self.electronMomentum)
        self.childMomentumMagnitude = np.linalg.norm(self.childMomentum)
        return self

