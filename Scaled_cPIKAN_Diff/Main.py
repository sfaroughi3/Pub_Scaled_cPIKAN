"""Scaled-cPIKANs: Spatial Variable and Residual Scaling in Chebyshev-based
   Physics-informed Kolmogorov-Arnold Networks

DEVELOPED AT:
                    Department of Chemical Engineering
                    University of Utah, Salt Lake City,
                    Utah 84112, USA
                    DIRECTOR: Prof.  Salah A Faroughi

DEVELOPED BY:
                    FARINAZ MOSTAJERAN

MIT License

Copyright (c) 2024 Farinaz Mostajeran

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import scipy.io as sio
import time
import numpy as np
import VS_MLP as mmain_MLP  # MLP-based PINN model
import VS_ChebKAN as mmain_cKAN  # Chebyshev-based KAN model
import Data_Diff as dataHH  # Data generation functions for the Diffusion equation

# ========= set device
if_gpu = True  # Use GPU if available
device = "cpu"
if if_gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.backends.mps.is_available():
    print("GPU is available on MAC")
    device = torch.device("mps")
print(device)
# ========= 

# Problem parameters
VS = 6.0  # Spatial scaling factor
code = 100  # Run identifier

epochs = 10000  # Number of training epochs
lr = 1.0e-3  # Learning rate

# Loss weights: [PDE residual, initial condition, boundary condition]
Weight_Loss = [0.01, 1.0, 1.0]  

# Diffusion equation parameter
Diff_param = 0.1

# Domain boundaries for x and t
x_interval = [-6.0, 6.0]
t_interval = [0.0, 1.0]
Num_Dom = 2000  # Number of collocation points
Num_bound = 200  # Number of boundary condition points
Num_initial = 400  # Number of initial condition points

# Model selection
#"""
Method = 'cKAN'   
Layers = 2  # Number of hidden layers
Neurons = 8  # Number of neurons per layer
degree_cKAN = 5  # Chebyshev polynomial degree
#"""

"""
Method = 'MLP'
Layers = 2    # Number of hidden layers
Neurons = 19  # Number of neurons per layer
"""

# Generate data: collocation, initial, and boundary points
X_int = dataHH.X_dom(Num_Dom, x_interval, t_interval)
X_0, U_0 = dataHH.Initial_Cond(Num_initial, VS, t_interval)
X_b, U_b = dataHH.Boundary_Cond(Num_bound, x_interval, t_interval) 

# Convert numpy arrays to PyTorch tensors and send to device
X_int = torch.from_numpy(X_int).float().to(device)
X_b = torch.from_numpy(X_b).float().to(device)
U_b = torch.from_numpy(U_b).float().to(device)
X_0 = torch.from_numpy(X_0).float().to(device)
U_0 = torch.from_numpy(U_0).float().to(device)

# Model training: choose cKAN or MLP
if Method == 'cKAN':
    #####==============================================================cKAN
    #####==============================================================
    #####==============================================================
    # cKAN model setup and training
    print(Method)
    Name_sio_cKAN = 'cPIKAN_'+str(code)+'_VS'+str(VS)+'_L'+str(Layers)+'_N'+str(Neurons)+'_D'+str(degree_cKAN)

    layer_widths_cKAN = [2] + Layers * [Neurons] + [1]  # Define network architecture
    chebnet = mmain_cKAN.ChebKAN_Network(layer_widths_cKAN, degree_cKAN, VS, Diff_param)
    chebnet.to(device)
    start_time = time.time()
    chebnet.fit(X_int, X_0, U_0, X_b, U_b,  epochs, Weight_Loss,
                lr,  nn.MSELoss(), 
                Name_sio_cKAN)
    chebnet.eval()
    elapsed_cKAN = time.time() - start_time
    
    loss_CL_cKAN = chebnet.Train_Loss

    # Save model checkpoint
    checkpoint_cKAN = {'layer_widths': chebnet.layer_widths,
                         'degree': chebnet.degree,
                         'VS': chebnet.VS, 
                         'Diff_param': chebnet.Diff_param,
                         'state_dict': chebnet.state_dict()}

    torch.save(checkpoint_cKAN, 'Checkpoint_'+Name_sio_cKAN+'.pth')

    # Save training results in MATLAB format
    sio.savemat(Name_sio_cKAN +'.mat', {'run_time':elapsed_cKAN, 'layer_widths':layer_widths_cKAN,
                                   'degree':degree_cKAN, 'device':str(device),
                                   'loss_CL':loss_CL_cKAN, 'lr':lr, 'Weight_Loss':Weight_Loss,
                                   'VS':VS, 'Num_initial':Num_initial,
                                   'Num_Dom':Num_Dom, 'Num_bound':Num_bound})
    del(chebnet)
    
elif Method == 'MLP':
    #####==============================================================MLP
    #####==============================================================
    #####==============================================================
    # MLP model setup and training
    print(Method)
    Name_sio_MLP = 'MLP_'+str(code)+'_VS'+str(VS)+'_L'+str(Layers)+'_N'+str(Neurons)
    layer_widths = [2] + Layers * [Neurons] + [1]  # Define network architecture
    mlpnet = mmain_MLP.MLP_Network(layer_widths, VS, Diff_param)
    mlpnet.to(device)
    start_time = time.time()
    mlpnet.fit(X_int, X_0, U_0, X_b, U_b,  epochs, Weight_Loss,
               lr,  nn.MSELoss(), Name_sio_MLP)
    mlpnet.eval()
    elapsed_MLP = time.time() - start_time
    
    loss_CL_MLP = mlpnet.Train_Loss

    # Save model checkpoint
    checkpoint_MLP = {'layer_widths': mlpnet.layer_widths,
                  'VS': mlpnet.VS,
                  'Diff_param': mlpnet.Diff_param,
                  'state_dict': mlpnet.state_dict()}
       
    torch.save(checkpoint_MLP, 'Checkpoint_'+Name_sio_MLP+'.pth')
        
    # Save training results in MATLAB format
    sio.savemat(Name_sio_MLP +'.mat', {'run_time':elapsed_MLP, 'layer_widths':layer_widths,
                                       'device':str(device),
                                       'loss_CL':loss_CL_MLP, 'Weight_Loss':Weight_Loss,
                                       'VS':VS, 'Num_initial':Num_initial,
                                       'Num_Dom':Num_Dom, 'Num_bound':Num_bound})
    del(mlpnet)
