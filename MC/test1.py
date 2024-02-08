#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 12:09:39 2024

@author: giacomomarocco
"""

from betaDecay import BetaDecay
import numpy as np


b = BetaDecay(2, 1,0.5,-0.2,0.3)
b.electronMomentum + b.childMomentum + b.neutrinoMomentum


