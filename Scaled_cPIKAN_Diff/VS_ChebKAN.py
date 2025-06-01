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
import time
#import numpy as np
from ChebyKANLayer import ChebyKANLayer_2

# Define the main ChebKAN network class
class ChebKAN_Network(nn.Module):
    
    #_#_===========================================================================
    def __init__(self, layer_widths, degree, VS, Diff_param):
        super(ChebKAN_Network, self).__init__()
        
        # Initialize network architecture parameters
        self.layer_widths = layer_widths  #List of layer sizes for the network
        self.degree = degree              # Chebyshev polynomial degree
        self.VS = VS                      # Spatial scaling factor
        self.Diff_param = Diff_param      # Diffusion equation parameter
        
        # Create a list to store Chebyshev layers
        self.Cheb_layers = nn.ModuleList()
        
        # Initialize each Chebyshev layer based on the provided widths
        for i in range(len(layer_widths) - 1):
            self.Cheb_layers.append(ChebyKANLayer_2(layer_widths[i], layer_widths[i+1], degree))
        
        # Store the training loss at each epoch
        self.Train_Loss = []
    #_#_===========================================================================    
    def forward(self, x):
        # Enable gradients for input x
        x = x.clone().detach().requires_grad_(True)
        
        out = x
        # Forward pass through all Chebyshev layers
        for i in range(len(self.layer_widths)-1):
            out = self.Cheb_layers[i](out)
        
        return out, x
    #_#_===========================================================================
    def gradient(self, y, x, grad_outputs=None):
        # Compute gradients dy/dx for each output dimension
        y_x = torch.zeros_like(y)
        y_y = torch.zeros_like(y)
        for i in range(y.shape[-1]):
            if grad_outputs is None:
                grad_outputs = torch.ones_like(y[:,i:i+1])
            grad = torch.autograd.grad(y[:,i:i+1], [x], grad_outputs=grad_outputs, create_graph=True)[0]
            y_x[:,i:i+1] = grad[...,0:1].reshape(y.shape[0],1)
            y_y[:,i:i+1] = grad[...,1:2].reshape(y.shape[0],1)
        
        return y_x, y_y
    #_#_===========================================================================
    def net_DiffEqu(self, X):
        # Compute the PDE residual: Diffusion equation U_t - alpha * U_xx = 0
        U, x = self.forward(X)
        U_x, U_t = self.gradient(U, x)
        U_xx, _ = self.gradient(U_x, x)
        
        # Scale the second derivative based on VS
        U_xx_VS = self.Diff_param * U_xx / (self.VS**2)
        
        # Compute the residual f = U_t - scaled U_xx
        f = U_t - U_xx_VS
        
        return U, f
    #_#_===========================================================================
    def net_U(self, X):
        # Return the predicted solution U at points X
        U, x = self.forward(X)
        return U
    #_#_===========================================================================
    def PDE_loss(self, X_int, X_0, U_0, X_b, U_b, loss_func, Weight_Loss):
        # Compute the total loss combining PDE residuals and boundary/initial conditions
        
        _, Res = self.net_DiffEqu(X_int)  # PDE residual at interior points
        U_b_pred = self.net_U(X_b)        # Predicted boundary values
        U_0_pred = self.net_U(X_0)        # Predicted initial condition
        
        # Zero vector for residual target
        f = torch.zeros(Res.shape[0], Res.shape[1], device=X_int.device)
        
        # Weighted sum of PDE residual loss, initial condition loss, and boundary loss
        loss = Weight_Loss[0] * loss_func(Res, f) + \
               Weight_Loss[1] * loss_func(U_0_pred, U_0) + \
               Weight_Loss[2] * loss_func(U_b_pred, U_b)
                   
        return loss
    #_#_===========================================================================
    def fit(self, X_int, X_0, U_0, X_b, U_b, epochs, Weight_Loss,
            lr, loss_func, Name_sio_cKAN):
        
        # Training loop using Adam optimizer
        start_time1 = time.time()
        
        self.train()
        optimiser = torch.optim.Adam(self.parameters(), lr=lr)
        epoch = 0
        while epoch < epochs:
            epoch += 1

            optimiser.zero_grad()
            train_loss = self.PDE_loss(X_int, X_0, U_0, X_b, U_b, loss_func, Weight_Loss)
            self.Train_Loss.append([1.1, train_loss.detach().cpu().numpy()])
            Check_iter_new = train_loss
                
            train_loss.backward()
            optimiser.step()
            
            # Print loss every 10 epochs
            if epoch % 10 == 0:    
                elapsed1 = time.time() - start_time1
                print('\r(cKAN) Epoch: %d, train-Loss: %.5E, runtime: %f' % (epoch, Check_iter_new, elapsed1))
                start_time1 = time.time()
            
            start_time1 = time.time()
    #_#_===========================================================================       
    def predict_u(self, x):
        # Predict solution U at given points x
        U = self.net_U(x)
        return U
