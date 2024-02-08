#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 14:17:12 2023

@author: giacomomarocco
"""

import numpy as np
import constants

mF = 0.
mGT = 1/3

print()
#The JTW coefficients, ignoring subleading electron mass supressed helicity flips and Coulomb corrections (to be done)
class JTW_Coefficients:
    def __init__(self, electronE, LHOnly = True):
        self.c1Vector = 1
        self.c1AxialVector = 1.27
        self.c1Tensor=0
        self.c1Scalar = 0
        if LHOnly == True:
            self.__SetLH()
        self.xi = abs(mF)**2*(abs(self.c1Vector)**2 + abs(self.c2Vector)**2 + abs(self.c1Scalar)**2 + abs(self.c2Scalar)**2)\
            + abs(mGT)**2*(abs(self.c1AxialVector)**2 + abs(self.c2AxialVector)**2 + abs(self.c1Tensor)**2 + abs(self.c2Tensor)**2)
        self.a = abs(mF)**2*(abs(self.c1Vector)**2 + abs(self.c2Vector)**2 - abs(self.c1Scalar)**2 - abs(self.c2Scalar)**2)\
        - abs(mGT)**2*(abs(self.c1AxialVector)**2 + abs(self.c2AxialVector)**2 - abs(self.c1Tensor)**2 - abs(self.c2Tensor)**2)/3
        
        self.b = 2*np.sqrt(1-constants.alphaEM**2*constants.protonNumber**2)*np.real(abs(mF)**2*(self.c1Vector * np.conjugate(self.c1Scalar) + self.c2Vector * np.conjugate(self.c2Scalar))\
        + abs(mGT)**2*(self.c1AxialVector * np.conjugate(self.c1Tensor) + self.c2AxialVector * np.conjugate(self.c2Tensor)))
            
        self.A = 2*np.real(np.abs(mGT)**2*(self.c1Tensor*np.conjugate(self.c2Tensor)-self.c1AxialVector*np.conjugate(self.c2AxialVector))\
                      +np.abs(mF)*np.abs(mGT)*(self.c1Scalar*np.conjugate(self.c2Tensor) + self.c2Scalar*np.conjugate(self.c1Tensor)\
                                               - self.c1Vector*np.conjugate(self.c2AxialVector) - self.c2Vector*np.conjugate(self.c1AxialVector)))
            
        self.B = 2*np.real(np.abs(mGT)**2*( (constants.electronMass/electronE)*(self.c1Tensor*np.conjugate(self.c2AxialVector) + self.c2Tensor*np.conjugate(self.c1AxialVector))\
                                      + self.c1Tensor*np.conjugate(self.c2Tensor) + self.c1AxialVector*np.conjugate(self.c2AxialVector))\
                      -np.abs(mGT)*np.abs(mF)*(self.c1Scalar*np.conjugate(self.c2Tensor) + self.c2Scalar*np.conjugate(self.c1Tensor)\
                                               + self.c1Vector*np.conjugate(self.c2AxialVector) + self.c2Vector*np.conjugate(self.c1AxialVector)\
                                                   + (constants.electronMass/electronE)*(self.c1Scalar*np.conjugate(self.c2AxialVector) + self.c2Scalar*np.conjugate(self.c1AxialVector)\
                                                    +self.c1Vector*np.conjugate(self.c2Tensor) + self.c2Vector*np.conjugate(self.c1Tensor))))
        
        self.D = 2*np.imag(np.abs(mF)*np.abs(mGT)*(self.c1Scalar*np.conjugate(self.c1Tensor) + self.c2Scalar*np.conjugate(self.c2Tensor)\
                                    - self.c1Vector*np.conjugate(self.c1AxialVector) - self.c2Vector*np.conjugate(self.c2AxialVector)))
            
    def __SetLH(self):
            self.c2Vector = self.c1Vector
            self.c2AxialVector = self.c1AxialVector
            self.c2Tensor = self.c1Tensor
            self.c2Scalar = self.c1Scalar
