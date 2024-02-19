#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:35:00 2024

@author: giacomomarocco
"""

def logLikelihood(data, pdfNewPhysics):
    #follows this outline https://arxiv.org/abs/1503.07622
    #data is an array of N-dimensional vectors, where N is the number of features of our dataset
    #the data is drawn from the background-only hypothesis
    #pdf is the probability density function for the hypothesis of background+signal -- it is a function of an N-dim vector
    
def generateToyData(experimentSize, pdfBackgroundOnly):
    for i in range(experimentSize):