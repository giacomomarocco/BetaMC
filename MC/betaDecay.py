#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 12:13:00 2023

@author: giacomomarocco
"""

#I should make sure that I'm getting the three-momentum of the daughter correctly.

import numpy as np
import scipy as sp
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
        self.protonError = 1e-2 # Uncertainty in sphere momentum of 1e-2 MeV
        self.angularError = 1e-2 # Uncertainty in angle of electron of 1e-2 rad
        self.electronError = 1e-2 # Uncertainty in electron energy of 1e-2 MeV
        self.unpolarised = unpolarised


        self.cosThetaElectronNu()   # Creates the angle between electron and neutrino
        self.neutrinoFourMomentum() # Create the neutrino 4-momentum
        self.electronFourMomentum() # Create the electron 4-momentum
        self.childMomentum()
        self.cosThetaElectronNucleon()
        self.weight()
        self.error()
        self.missingMomentum()
        self.missingEnergy()
    
    def cosThetaElectronNu(self):
        # Return cosine of angle between electron and neutrino
        self.electronDirection = direction_from_polar(self.electronCosTheta, self.electronPhi)
        self.neutrinoDirection = direction_from_polar(self.neutrinoCosTheta, self.neutrinoPhi)
        self.cosThetaElectronNu = np.dot(self.electronDirection, self.neutrinoDirection)
        return self
    
    
    def nucleusAngularMomentum(self):
        if self.unpolarised:
            self.angularMomentumDirection = direction_from_polar(np.random.uniform(-1,1),np.random.uniform(0,2*np.pi))
        else:
            self.angularMomentumDirection = np.array([0,0,1])
        return self
    
    def neutrinoEnergy(self):
        # Define the neutrino energy given the electron's energy and the Q-value, neglecting the small kinetic energy of child nucleus
        self.neutrinoEnergy = self.QValue - (self.electronEnergy - electronMass)
        return self
    
    def neutrinoFourMomentum(self):
        self.neutrinoEnergy()
        self.neutrinoMomentum = Momentum4.from_polar(self.neutrinoEnergy, self.neutrinoCosTheta, self.neutrinoPhi , 0).threeMomentum
        return self
    
    def electronFourMomentum(self):
        electron4Momentum = Momentum4.from_polar(self.electronEnergy, self.electronCosTheta, self.electronPhi, electronMass)
        self.electronMomentum = electron4Momentum.threeMomentum
        self.electronMomentumMagnitude = electron4Momentum.momentumMag
        if self.electronMomentumMagnitude == 0:
            raise ValueError("electron has no momentum")
        return self
    
    def childMomentum(self):
        self.childMomentum = -self.electronMomentum - self.neutrinoMomentum 
        self.childMomentumMagnitude = np.linalg.norm(self.childMomentum)
        if self.childMomentumMagnitude == 0:
            raise ValueError("child has no momentum")
        return self

    def cosThetaElectronNucleon(self):
        self.cosThetaElectronNucleon = np.dot(self.electronMomentum,self.childMomentum)/(self.electronMomentumMagnitude*self.childMomentumMagnitude)
        if self.cosThetaElectronNucleon > 1:
            raise ValueError("cosThetaElectronNucleon > 1")
        if self.cosThetaElectronNucleon < -1:
            self.cosThetaElectronNucleon = -1
        return self

    def missingEnergy(self):
        self.missingEnergy = - (self.electronEnergy - self.QValue - electronMass)
        self.missingMassSquared = self.missingEnergy**2 - self.missingMomentumMagnitude**2
        return self

    def weight(self):
        # Define the weight of the event through the particular probability distribution
        self.jtw = JTW_Coefficients(self.electronEnergy)
        self.FermiFunction = self.FermiFunction()
        self.phaseSpaceFactor = self.phaseSpaceFactor()
        self.nucleusAngularMomentum()
        aFactor = (self.electronMomentumMagnitude*self.cosThetaElectronNu/self.electronEnergy)
        bFactor = electronMass/self.electronEnergy
        AFactor = np.dot(self.angularMomentumDirection, self.electronDirection)*self.electronMomentumMagnitude/self.electronEnergy
        BFactor = np.dot(self.angularMomentumDirection, self.neutrinoDirection)
        DFactor = np.dot(self.angularMomentumDirection,np.cross(self.electronDirection,self.neutrinoDirection))*self.electronMomentumMagnitude/self.electronEnergy
        self.weight = self.FermiFunction*self.phaseSpaceFactor\
            *(self.jtw.xi + self.jtw.a * aFactor + self.jtw.b * bFactor + self.jtw.A * AFactor + self.jtw.B * BFactor + self.jtw.D * DFactor)
        return self
    
    def error(self):
        prefactor = 1/(1+(self.electronMomentumMagnitude/self.childMomentumMagnitude)**2)
        firstFactor = (self.childMomentumMagnitude - self.electronMomentumMagnitude*self.cosThetaElectronNucleon)/(self.electronMomentumMagnitude**2+self.childMomentumMagnitude**2)
        secondFactor = 1
        self.error =  prefactor*(firstFactor**2*(self.electronError**2 + (self.electronMomentumMagnitude**2/(self.childMomentumMagnitude**2))*self.protonError**2) + self.angularError**2*np.sqrt(1-self.cosThetaElectronNucleon**2) )
        return self
            
    def FermiFunction(self):
        alpha = 1/137
        Z = 10
        S = np.sqrt(1-alpha**2 * Z**2)
        rho = 1.2 * Z**(1/3)
        eta = Z*0.3**2*self.electronEnergy/self.electronMomentumMagnitude
        fraction = 2*(1+S)/((sp.special.gamma(1+2*S))**2)
        firstTerm = (2*self.electronMomentumMagnitude*rho)**(2*S-2)
        secondTerm = np.exp(np.pi*eta)
        thirdTerm = np.absolute(sp.special.gamma(S+eta*1j))**2
        return fraction*firstTerm*secondTerm*thirdTerm
    
    def phaseSpaceFactor(self):
        return self.electronMomentumMagnitude*self.electronEnergy*(electronMass+self.QValue-self.electronEnergy)**2
    
    def missingMomentum(self):
        self.missingMomentum = -(self.childMomentum + self.electronMomentum)
        self.missingMomentumMagnitude = np.linalg.norm(self.missingMomentum)
        return self
    
    
    