#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:09:39 2024

@author: giacomomarocco
"""

import time
from utils import generate_samples
import numpy as np
import scipy as sp
from betaDecay import BetaDecay
from gammaDecay import GammaDecay
import matplotlib.pyplot as plt
from constants import electronMass




class test:
    def __init__(self, QValue, num_samples, decayType, gammaEnergy = 0 ):
        start = time.time()
        self.QValue = QValue
        self.gammaEnergy = gammaEnergy
        self.num_samples = num_samples
        self.decayType = decayType
        self.__getSamples()
        
        self.missingMomentum = np.empty(self.num_samples)
        self.missingEnergy = np.empty(self.num_samples)
        self.betaEnergy = np.empty(self.num_samples)
        self.weights = np.empty(self.num_samples)
        self.missingMassSquared = np.empty(self.num_samples)
        
        for i in range(num_samples):
            if self.decayType == 'beta':
                decay = BetaDecay(self.QValue, self.samples[0][i], self.samples[1][i], self.samples[2][i], self.samples[3][i])
                self.missingEnergy[i], self.missingMomentum[i], self.betaEnergy[i], self.weights[i] = [decay.missingEnergy, decay.missingMomentumMagnitude, decay.electronEnergy, decay.weight]
                self.missingMassSquared[i] = decay.missingMassSquared
            elif self.decayType == 'gamma':
                decay = GammaDecay(self.gammaEnergy, self.QValue, self.samples[0][i], self.samples[1][i], self.samples[2][i], self.samples[3][i])
                self.missingEnergy[i], self.missingMomentum[i], self.betaEnergy[i], self.weights[i] = [decay.missingEnergy, decay.missingMomentumMagnitude, decay.electronEnergy, decay.weight]
                self.missingMassSquared[i] = decay.missingMassSquared
        end = time.time()
        self.reweightedWeights = self.weights/sum(self.weights)
        self.weights = self.reweightedWeights
        print(end - start)



    def __getSamples(self):
        #Only randomise electron cosTheta; co-ordinate system aligned so that phi vanishes
        if self.decayType == 'beta':
            self.electronE = np.linspace(electronMass+1e-3, self.QValue-1e-3 + electronMass, 200)
        elif self.decayType == 'gamma':
            self.electronE = np.linspace(electronMass+1e-3, self.QValue-1e-3 - self.gammaEnergy + electronMass, 100)
        self.electronCosTheta = np.linspace(-1, 1, 100)    
        self.nuCosTheta = np.linspace(-1, 1, 100)    
        self.nuPhi = np.linspace(0,2 * np.pi,100)
        self.samples = np.transpose(generate_samples(self.electronE, self.electronCosTheta,self.nuCosTheta,self.nuPhi, n_samples = self.num_samples))

    def hist2d(self):
        betaKineticEnergy = self.betaEnergy-electronMass
        
        plt.hist2d(self.missingEnergy,self.missingMomentum , weights=self.weights, bins = 100, range = [[0, self.QValue],[0, self.QValue]])
        plt.ylabel(r'$|\mathbf{p}_\mathrm{miss} =  - \mathbf{p}_\beta - \mathbf{p}_\mathrm{sp}|$', fontsize = 'large')
        plt.xlabel(r'$E_\mathrm{miss} = Q_\mathrm{EC} - T_\beta$', fontsize = 'large')
        plt.show()
        
    def plotMissingMass(self):
        missingMass = self.missingEnergy**2 - self.missingMomentum**2
        plt.hist(missingMass, bins = 50, weights = self.weights, density = True, range = [0, self.QValue**2])
        plt.xlabel(r'$m_\mathrm{miss}^2$')
        plt.show()
        return self
        
    def inferredMomentum(self, electronMomentum):
        return self
        
        


b = test(2,500000, 'beta')
g = test(2,500000, 'gamma', 1.5)
b.hist2d()
g.hist2d()
plt.hist2d(g.missingMassSquared, g.missingMomentum,bins = 100, weights = g.weights, range = [[0,3],[0,2]])
plt.show()
plt.hist2d(b.missingMassSquared, b.missingMomentum,bins = 100, weights = b.weights, range = [[0,3],[0,2]])
plt.show()
plt.hist(np.abs(b.missingMassSquared)**(1/2), weights = g.weights, bins = 100, range = [0,2])
plt.show()
plt.hist(np.abs(g.missingMassSquared)**(1/2), weights = g.weights, bins = 100, range = [0,2])
plt.show()
# b.plotMissingMass()
# g.plotMissingMass()

