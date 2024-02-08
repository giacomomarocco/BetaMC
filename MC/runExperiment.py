#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 13:09:16 2023

@author: giacomomarocco
"""
import time

from utils import generate_samples
import numpy as np
import scipy as sp
from betaDecay import BetaDecay
import matplotlib.pyplot as plt
from constants import *
from math import log10, floor

class runExperiment:
    def __init__(self,QValue, num_samples):
        start = time.time()

        self.QValue = QValue
        self.num_samples = num_samples
        self.__getSamples()
        self.cosTheta = np.empty(self.num_samples)
        self.weights = np.empty(self.num_samples)
        self.errors = np.empty(self.num_samples)
        for i in range(num_samples):
            betaDecay = BetaDecay(self.QValue, self.samples[0][i], self.samples[1][i], self.samples[2][i], self.samples[3][i])
            self.cosTheta[i], self.weights[i], self.errors[i] = [betaDecay.getCosTheta(), betaDecay.getWeight(), betaDecay.getError()]
        end = time.time()
        self.reweightedWeights = self.num_samples*self.weights/sum(self.weights)
        print(end - start)

    def plotHistogram(self):
        plt.hist(self.cosTheta, 100, weights = self.weights, density = True)
        plt.show()
        
    def round_to_1(self, x):
        return (round(x, -int(floor(log10(abs(x))))), -int(floor(log10(abs(x)))))

    def plot(self, linear_fit = True):
        # plt.rcParams['text.usetex'] = True
        plt.rcParams['figure.dpi'] = 250
        [number, bin_edges] = np.histogram(self.cosTheta, 50, weights = self.weights)
        [numberDensity, bin_edges] = np.histogram(self.cosTheta, 50, weights = self.weights, density=True)
        xValues = [(bin_edges[i+1]+bin_edges[i])/2 for i in range(len(number))]
        fig, ax = plt.subplots()
        plt.errorbar(xValues, numberDensity, yerr = numberDensity/np.sqrt(number), xerr = 1e-2/np.sqrt(number) , label = 'Simulated data')
        plt.xlabel(r'$\mathrm{cos} \,\theta_{e\nu}$', fontsize = 'large')
        plt.ylabel(r'$\frac{\mathrm{d}\Gamma}{\mathrm{d} \cos \, \theta_{e\nu}}$', rotation='horizontal',fontsize = 'x-large', labelpad = 10)
        if linear_fit:
            res = sp.stats.linregress(xValues, numberDensity)
        plt.plot(xValues, res.intercept + res.slope*np.array(xValues), 'r', label = 'Best fit')
        [error, sigFig] = self.round_to_1(res.stderr)
        # plt.text(0.99, 0.99, r'$\mathrm{slope} = {} \pm {}$'.format(res.slope, res.stderr),ha='right', va='top')
        ax.text(0.75, 0.75, r'$a_{\beta \nu} = $'+'{}'.format(round(res.slope,sigFig)) + r'$ \pm$'+'{}'.format(error), horizontalalignment='right',
        verticalalignment='top',
        transform=ax.transAxes)
        plt.legend()
        plt.show()
        
    def __getSamples(self):
        #Only randomise cosTheta; co-ordinate system aligned so that phi vanishes
        self.electronE = np.linspace(electronMass, self.QValue + electronMass, 100)
        self.electronCosTheta = np.linspace(-1, 1, 100)    
        self.nuCosTheta = np.linspace(-1, 1, 100)    
        self.nuPhi = np.linspace(0,2 * np.pi,100)
        self.samples = np.transpose(generate_samples(self.electronE, self.electronCosTheta,self.nuCosTheta,self.nuPhi, n_samples = self.num_samples))


    