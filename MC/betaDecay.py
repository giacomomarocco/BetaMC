#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:13:00 2023

@author: giacomomarocco
"""

#I should make sure that I'm getting the three-momentum of the daughter correctly.

import numpy as np
from constants import electronMass
from constants import protonMass
from utils import direction_from_polar
from utils import momentumMagnitude
from momentum4 import Momentum4
from JTWParameters import JTW_Coefficients

class BetaDecay:
    def __init__(self, QValue, electronEnergy, electronCosTheta, neutrinoCosTheta, neutrinoPhi, unpolarised = True):
        # A particular beta decay event is defined by the energy released in the process in addition to the five other kinematic variables
       
        self.QValue = QValue
        self.electronEnergy = electronEnergy
        self.electronMomentum = momentumMagnitude(self.electronEnergy, electronMass)
        self.electronCosTheta = electronCosTheta
        self.electronPhi = 0. # The fifth variable, the electron azimuthal angle, is set to zero in my coordinate system
        self.neutrinoCosTheta = neutrinoCosTheta
        self.neutrinoPhi = neutrinoPhi
        self.protonError = 1e-2
        self.angularError = 1e-2
        self.electronError = 1e-2

        self.cosThetaElectronNu()
        self.unpolarised = unpolarised
        
    def getWeight(self):
        self.setWeight()
        return self.weight
    
    def getError(self):
        self.setError()
        return self.error
    
    def getCosTheta(self):
        return self.cosThetaElectronNu

    def cosThetaElectronNu(self):
        # Return cosine of angle between electron and neutrino
        self.electronDirection = direction_from_polar(self.electronCosTheta, self.electronPhi)
        self.neutrinoDirection = direction_from_polar(self.neutrinoCosTheta, self.neutrinoPhi)
        self.cosThetaElectronNu = np.dot(self.electronDirection, self.neutrinoDirection)
        return self
    def cosThetaElectronNucleon(self):
        self.setProtonFourMomentum()
        self.cosThetaElectronNucleon = (-np.sqrt(self.electronMomentum**2+self.protonMomentum**2)*self.cosThetaElectronNu - self.electronMomentum)/self.protonMomentum
        if self.cosThetaElectronNucleon > 1:
            self.cosThetaElectronNucleon = 0.99
        if self.cosThetaElectronNucleon < -1:
            self.cosThetaElectronNucleon = -0.99
        return self
    def nucleusAngularMomentum(self):
        if self.unpolarised:
            self.angularMomentumDirection = direction_from_polar(np.random.uniform(-1,1),np.random.uniform(0,2*np.pi))
        else:
            self.angularMomentumDirection = np.array([0,0,1])
        return self
    
    def neutrinoEnergy(self):
        # Define the neutrino energy given the electron's energy and cosThetaElectronNu
        alpha = self.electronMomentum*self.cosThetaElectronNu+self.QValue + electronMass-self.electronEnergy
        beta = self.electronMomentum**2*(1-self.cosThetaElectronNu**2)/protonMass**2
        gamma = self.electronMomentum*self.cosThetaElectronNu+protonMass
        self.neutrinoEnergy = protonMass*(1+2*alpha/protonMass-beta)**2-gamma
        return self
        
    def setWeight(self):
        # Define the weight of the event through the particular probability distribution
        self.jtw = JTW_Coefficients(self.electronEnergy)
        self.FermiFunction = self.FermiFunction()
        self.phaseSpaceFactor = self.phaseSpaceFactor()
        self.nucleusAngularMomentum()
        aFactor = (self.electronMomentum*self.cosThetaElectronNu/self.electronEnergy)
        bFactor = electronMass/self.electronEnergy
        AFactor = np.dot(self.angularMomentumDirection, self.electronDirection)*self.electronMomentum/self.electronEnergy
        BFactor = np.dot(self.angularMomentumDirection, self.neutrinoDirection)
        DFactor = np.dot(self.angularMomentumDirection,np.cross(self.electronDirection,self.neutrinoDirection))*self.electronMomentum/self.electronEnergy
        self.weight = self.FermiFunction*self.phaseSpaceFactor\
            *(self.jtw.xi + self.jtw.a * aFactor + self.jtw.b * bFactor + self.jtw.A * AFactor + self.jtw.B * BFactor + self.jtw.D * DFactor)
        return self
    
    def setError(self):
        self.cosThetaElectronNucleon()
        prefactor = 1/(1+(self.electronMomentum/self.protonMomentum)**2)
        firstFactor = (self.protonMomentum - self.electronMomentum*self.cosThetaElectronNucleon)/(self.electronMomentum**2+self.protonMomentum**2)
        secondFactor = 1
        self.error =  prefactor*(firstFactor**2*(self.electronError**2 + (self.electronMomentum**2/(self.protonMomentum**2))*self.protonError**2) + self.angularError**2*np.sqrt(1-self.cosThetaElectronNucleon**2) )
        return self
    
    def setNeutrinoFourMomentum(self):
        self.neutrinoEnergy()
        self.neutrino4Momentum = Momentum4.from_polar(self.neutrinoEnergy, self.neutrinoCosTheta, self.neutrinoPhi , 0)
    
    def setElectronFourMomentum(self):
        self.electron4Momentum = Momentum4.from_polar(self.electronEnergy, self.electronCosTheta, self.electronPhi, electronMass)
        
    def setProtonFourMomentum(self):
        self.setElectronFourMomentum()
        self.setNeutrinoFourMomentum()
        self.Proton4Momentum = Momentum4.from_threeMomentum(-(self.electron4Momentum.p+self.neutrino4Momentum.p), protonMass)
        self.protonMomentum = momentumMagnitude(self.Proton4Momentum.p[0], protonMass)

    
    def getProtonFourMomentum(self):
        self.setProtonFourMomentum()
        return self.Proton4Momentum
    
    def getProtonMomentum(self):
        self.setProtonFourMomentum()
        return  self.protonMomentum
    
    def getElectronFourMomentum(self):
        self.setElectronFourMomentum()
        return self.electron4Momentum
    
    def getNeutrinoFourMomentum(self):
        self.setNeutrinoFourMomentum()
        return self.neutrino4Momentum

    def FermiFunction(self):
        return 1
    
    def phaseSpaceFactor(self):
        return self.electronMomentum*self.electronEnergy*(self.QValue-self.electronEnergy)**2
    
    
    
    
    