#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:27:40 2023

@author: giacomomarocco
"""

# test = runExperiment(50,10000000)
# test1 = test
from runExperiment import runExperiment
test = runExperiment(0.5,100000)
import numpy as np 
def round_to_1( x):
    return (round(x, -int(floor(log10(abs(x))))), -int(floor(log10(abs(x)))))



[number, bin_edges] = np.histogram(test.cosTheta, 50, weights = test.reweightedWeights, range = (-1.0, 0.99))
xValues = [(bin_edges[i+1]+bin_edges[i])/2 for i in range(len(number))]
plt.rcParams['figure.dpi'] = 300
fig, axs = plt.subplots(2, 1, height_ratios = [3,1])
axs[0].errorbar(xValues, number, yerr = 2*np.sqrt(number), xerr = 1e-2 , label = 'Simulated data', marker = ',', linestyle='None',)
bestFit = sp.stats.linregress(xValues, number)
axs[0].plot(xValues, bestFit.intercept + bestFit.slope*np.array(xValues), 'r', label = 'Best fit')
residuals = number - (bestFit.intercept + bestFit.slope*np.array(xValues))
interceptError = bestFit.intercept_stderr
gradientError = bestFit.stderr
def yPredictedError(x):
    return np.array([np.sqrt(x[i]**2*gradientError**2 + interceptError**2 + bestFit.slope**2*1e-4) for i in range(len(x))])
residualErrors = np.sqrt(number + yPredictedError(xValues)**2)
[error, sigFig] = round_to_1(50*bestFit.stderr/sum(number))

axs[0].text(0.9, 0.8, r'$a_{\beta \nu} = $'+'{}'.format(bestFit.slope) + r'$ \pm$'+'{}'.format(bestFit.stderr) + '\n'+ r'$ |C_T/C_A|^2 < 3 \times 10^{-3}$', horizontalalignment='right',
verticalalignment='top',
transform=axs[0].transAxes)

axs[1].set_xlabel(r'$\mathrm{cos} \,\theta_{e\nu}$', fontsize = 'large')
axs[0].set_ylabel(r'$\mathrm{Count}$',fontsize = 'x-large', labelpad = 10)
axs[1].set_ylabel(r'$\mathrm{Residual}/\sigma$',fontsize = 'large', labelpad = 10)



# axs[1].plot(xValues, residuals/residualErrors, marker = '.', linestyle='None')
axs[1].plot(xValues, residuals/(2*residualErrors), marker = '.', linestyle='None')
axs[1].set_ylim([-3,3])
axs[1].set_yticks(range(-2,3))
axs[1].grid(axis = 'y', linestyle = '--')
print(50*bestFit.slope/sum(number))
print(50*bestFit.stderr/sum(number))
plt.show()