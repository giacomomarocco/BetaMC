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
    def __init__(self, gammaEnergies, QValue, electronEnergy, electronCosTheta, neutrinoCosTheta, neutrinoPhi, photonMass = 0, unpolarised = True):
        #Initialise the beta decay, but now the energy available for beta decay is less (QValue-gammaEnergy).
        self.gammaEnergies = gammaEnergies
        self.totalGammaEnergy = sum(self.gammaEnergies)
        self.photonMass = photonMass
        super().__init__(QValue-self.totalGammaEnergy, electronEnergy, electronCosTheta, neutrinoCosTheta, neutrinoPhi, unpolarised = True)
        
    def gammaMomentum(self, gammaEnergy):
        cosThetaGamma = np.random.uniform(-1.0,1.0)
        phiGamma = np.random.uniform()*2*np.pi
        gammaMomentum = Momentum4.from_polar(gammaEnergy, cosThetaGamma, phiGamma, self.photonMass).threeMomentum
        return gammaMomentum
    
    def childMomentum(self):
        childMomentumBeforeGamma = -(self.neutrinoMomentum + self.electronMomentum)
        totalGammaMomentum = np.zeros(3)
        for gammaEnergy in self.gammaEnergies:
            totalGammaMomentum += self.gammaMomentum(gammaEnergy) 
        self.childMomentum = childMomentumBeforeGamma - totalGammaMomentum
        self.childMomentumMagnitude = np.linalg.norm(self.childMomentum)
        return self

    def missingEnergy(self):
        self.missingEnergy = - (self.electronEnergy - self.QValue - self.totalGammaEnergy- electronMass)
        self.missingEnergy = self.QValue + self.totalGammaEnergy - (self.electronEnergy - electronMass)
        self.missingMassSquared = self.missingEnergy**2 - self.missingMomentumMagnitude**2
        return self
