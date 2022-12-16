# FastPricing
This project aims to investigate the performance of machine learning methods for pricing financial derivatives.  
Given the competitiveness of a market-making environment, the ability to speedily quote option prices consistent with an ever-changing market environment is essential. Thus, the smallest acceleration or improvement over traditional pricing methods is crucial to avoid arbitrage. One can arrive at pricing speed-ups of several orders of magnitude by deploying machine learning techniques based on Gaussian process regression (GPR). However, this speed-up is obtained with a certain loss of accuracy. An essential focus of this project is a study of the speed-accuracy trade-off for pricing of vanilla (and exotic) options under different models such as i) Gaussian process regression, ii) random forests, and iii) deep neural networks.

## Create dataset
'HestonFFT.py'

## Gaussian Process Regression
'ParallelGPR.py'
'GreeksGPR.py'

## Neural Network
'NN.py'

## Utils
'TestPerformance.py'
'optim.py'

## Notebooks
'ParallelGPR.ipynb'
'NeuralNet.ipynb'
