# FastPricing
This project aims to investigate the performance of machine learning methods for pricing financial derivatives. 
Given the competitiveness of a market-making environment, the ability to speedily quote option prices consistent with an ever-changing market environment is essential. Thus, the smallest acceleration or improvement over traditional pricing methods is crucial to avoid arbitrage One can arrive at pricing speed-ups of several orders of magnitude by deploying machine learning techniques based on Gaussian Process Regression (GPR). However, this speed-up is obtained with a certain loss of accuracy. 
The essential focus of this project is the study of the speed-up accuracy trade-off for pricing vanilla European options under different models such as Gaussian Process Regression, Random Forest, and Deep Neural Networks (Task A). Additionally, the project studies how incorporating information about options Greeks could lead to improvements in the pricing accuracy for the GPR model (Task B).

Datasets and model weights are available at: https://drive.google.com/drive/folders/1f1FzBFMSWmIym31mLibrDB1_qywd_xdq?usp=sharing

### Team
- Giovanni La Cagnina
- Francesco Pettenon
- Nicolò Taroni

## Create dataset
- `HestonFFT.py`: file containing functions to obtain call option prices using Heston model via Fast Fourier Transform.
- `Simulation.py`: file containng functions to create grids of parameters and simulate call option prices.

## Gaussian Process Regression
- `FitGPR.py`: file containing functions to fit, save a timing a GPR model.
- `ParallelGPR.py`: file containing the implementation of the class ParallelGPR.
- `GreeksGPR.py`: file containing implementation of the class DeltaGPR and RhoGPR.
- `FitGreeksGPR.py`: file containig functions to fit, save and timing GreeksGPR models.

## Neural Network
- `ModelDeep.py`: file contating implementation of Neural Network Deep.|
- `ModelLarge.py`: file contating implementation of Neural Network Deep.|

## Utils
- `TestPerformance.py`
- `VisualizeError.py`
- `PlotUtils.py`
- `optim.py`

## Greeks
- `Greeks.py`
- `GreeksFFT.m`


## Notebooks
- `SimulationHeston.ipynb`
- `MeasureSimulationTime.ipynb`
- `GPR.ipynb`
- `ParallelGPR.ipynb`
- `NeuralNetLarge.ipynb`
- `NeuralNetDeep.ipynb`
- `RandomForest.ipynb`
- `VisualizeError.ipynb`
- `Greeks.ipynb`
- `GreeksGPR.ipynb`
- `AlphaValidation.ipynb`


