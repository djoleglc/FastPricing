import numpy as np
import torch 
import time 

def test_performanceGPR(X_, y_, model, type_="mean", to_return=False):
    """
    Function to evaluate the perfomance of a GPR based on a test set 
    - X_ : numpy array containing the feature of the test set 
    - y_ : numpy array containing the respose of the test set 
    - model : sklearn or similar model object used to estimate the prices 
    - type_: string variable describing which accuracy measure to estimate
    - to_return : boolean variable describing if it is necessary to return the accuracy measure neeeded
    
    """
    true = y_.reshape(-1, 1)
    s = time.time()
    pred = model.predict(X_).reshape(-1, 1)
    e = time.time()
    absolute_error = np.absolute(pred - true)
    
    print(f"Time:  {e-s}")
    if type_ == "mean":
        AAE = absolute_error.mean()
        print(f"AAE:  {AAE:.2e}")
        if to_return:
            return AAE
    if type_ == "max":
        MAE = absolute_error.max()
        print(f"MAE:  {MAE:.2e}")
        if to_return:
            return MAE
    if type_ == "both":
        AAE = absolute_error.mean()
        MAE = absolute_error.max()
        print(f"AAE:  {AAE:.2e}")
        print(f"MAE:  {MAE:.2e}")
        if to_return:
            return AAE, MAE

        
        
def test_performanceNN(X_, y_, model, type_ = "mean", to_return = False, time_ = False):
    """
     Function to evaluate the perfomance of a GPR based on a test set 
    - X_ : torch tensor containing the feature of the test set 
    - y_ : torch tensor containing the respose of the test set 
    - model : torch model object object used to estimate the prices 
    - type_: string variable describing which accuracy measure to estimate
    - to_return : boolean variable describing if it is necessary to return the accuracy measure neeeded
    - time_ : boolean variable describing if it is necessary to print the time 
    
    """
    model.eval()
    X_ =   X_.double() 
    true = y_.flatten()
    start = time.time()
    pred = model.forward( X_ ).flatten()
    end = time.time()
    if time_:
        print(f"Time:  {end-start}")
    model.train()
    absolute_error = torch.absolute(pred - true)
    if type_ == "mean":
        AAE = torch.mean(absolute_error).item()
        print(f"AAE:  {AAE:.2e}")
        if to_return:
            return AAE
    if type_ == "max":
        MAE = torch.max(absolute_error).item()
        print(f"MAE:  {MAE:.2e}")
        if to_return:
            return MAE
    if type_ == "both":
        AAE = torch.mean(absolute_error).item()
        MAE = torch.max(absolute_error).item()
        print(f"AAE:  {AAE:.2e}")
        print(f"MAE:  {MAE:.2e}")
        if to_return:
            return AAE, MAE
        