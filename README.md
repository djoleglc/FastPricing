# FastPricing
This project aims to investigate the performance of machine learning methods for pricing financial derivatives. 
Given the competitiveness of a market-making environment, the ability to speedily quote option prices consistent with an ever-changing market environment is essential. Thus, the smallest acceleration or improvement over traditional pricing methods is crucial to avoid arbitrage One can arrive at pricing speed-ups of several orders of magnitude by deploying machine learning techniques based on Gaussian Process Regression (GPR). However, this speed-up is obtained with a certain loss of accuracy. 
The essential focus of this project is the study of the speed-up accuracy trade-off for pricing vanilla European options under different models such as Gaussian Process Regression, Random Forest, and Deep Neural Networks (Task A). Additionally, the project studies how incorporating information about options Greeks could lead to improvements in the pricing accuracy for the GPR model (Task B).

Datasets and model weights are available at: https://drive.google.com/drive/folders/1f1FzBFMSWmIym31mLibrDB1_qywd_xdq?usp=sharing

### Team
- Giovanni La Cagnina: giovanni.lacagnina@epfl.ch
- Francesco Pettenon: francesco.pettenon@epfl.ch
- Nicolò Taroni: nicolo.taroni@epfl.ch

### Supervisors
- Urban Urlych: urban.urlych@epfl.ch
- Puneet Pasricha: puneet.pasricha@epfl.ch

## Create dataset
- `HestonFFT.py`: file containing functions to obtain call option prices using Heston model via Fast Fourier Transform.
- `Simulation.py`: file containing functions to create grids of parameters and simulate call option prices.

## Gaussian Process Regression
- `FitGPR.py`: file containing functions to fit, save a timing a GPR model.
- `ParallelGPR.py`: file containing the implementation of the class ParallelGPR.
- `GreeksGPR.py`: file containing implementation of the class DeltaGPR and RhoGPR.
- `FitGreeksGPR.py`: file containing functions to fit, save and timing GreeksGPR models.

## Neural Network
- `ModelDeep.py`: file containing implementation of Neural Network Deep.
- `ModelLarge.py`: file containing implementation of Neural Network Large.

## Utils
- `TestPerformance.py`: file containing functions to test the performances of implemented models.
- `VisualizeError.py`: file containing functions to visualize the errors of the model with respect to the feature used.
- `PlotUtils.py`: file containing functions to visualize the Greeks surfaces.
- `optim.py`: file containing the optimizer used for GPR models and for GreeksGPR models to minimize the log-likelihood.

## Greeks
- `Greeks.py`: file containing functions used to implemente the calculation of the Greeks using GPR, and Neural Networks model.
- `GreeksFFT.m`: Matlab file containing calculation of the Greeks values using Fast Fourier algorithm.


## Notebooks
- `SimulationHeston.ipynb`: notebook containing example of simulation of the data.
- `MeasureSimulationTime.ipynb`: notebook containing tests regarding the timing of the Fast Fourier method to obtain call option prices.
- `GPR.ipynb`: notebook containing fitting and performance of GPR models.
- `ParallelGPR.ipynb`: notebook containing fitting and performance of a ParallelGPR model.
- `NeuralNetLarge.ipynb`: notebook containing fitting and performance of a Neural Network Large model. 
- `NeuralNetDeep.ipynb`: notebook containing fitting and performance of a Neural Network Deep model. 
- `RandomForest.ipynb`: notebook containing fitting and performance of a Random Forest model. 
- `VisualizeError.ipynb`: notebook containing the visualization of the errors of different models with respect to the features.
- `Greeks.ipynb`: notebook containing calculation of Greeks, accuracy of the Greeks and Greeks surfaces for different models. 
- `GreeksGPR.ipynb`: notebook containing fitting and performance of DeltaGPR and RhoGPR.
- `AlphaValidation.ipynb`: notebook containing example of optimization of RhoGPR, DeltaGPR, and GPR models using a validation set to choose the regularization hyperparameter, called alpha.


