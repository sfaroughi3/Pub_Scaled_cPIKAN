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
import numpy as np
import VS_MLP as mmain_MLP
import VS_ChebKAN as mmain_cKAN

#_#_===========================================================================
def load_checkpoint_MLP(filename):
    """
    Load a pre-trained MLP model checkpoint from a file.
    
    Args:
        filename (str): Path to the checkpoint file.
    
    Returns:
        n_pre: Loaded MLP model.
        checkpoint['VS']: Scaling parameter used during training.
        checkpoint['Diff_param']: Diffusion equation parameters used during training.
    """
    checkpoint = torch.load(filename, map_location=torch.device('cpu'), weights_only=False)
    
    # Initialize the MLP model architecture using parameters from checkpoint
    n_pre = mmain_MLP.MLP_Network(checkpoint['layer_widths'], 
                        checkpoint['VS'],
                        checkpoint['Diff_param'])
    
    # Load the saved model weights into the MLP model
    n_pre.load_state_dict(checkpoint['state_dict'])
    
    return n_pre, checkpoint['VS'], checkpoint['Diff_param']

#_#_===========================================================================
def load_checkpoint_cKAN(filename):
    """
    Load a pre-trained Chebyshev-KAN model checkpoint from a file.
    
    Args:
        filename (str): Path to the checkpoint file.
    
    Returns:
        n_pre: Loaded Chebyshev-KAN model.
        checkpoint['VS']: Scaling parameter used during training.
        checkpoint['Diff_param']: Diffusion equation parameters used during training.
    """
    # Load the checkpoint dictionary containing model weights and parameters
    checkpoint = torch.load(filename, map_location=torch.device('cpu'), weights_only=False)
    
    # Initialize the Chebyshev-KAN model architecture using checkpoint parameters
    n_pre = mmain_cKAN.ChebKAN_Network(checkpoint['layer_widths'], 
                            checkpoint['degree'],
                            checkpoint['VS'],
                            checkpoint['Diff_param'])
    
    # Load the saved model weights into the Chebyshev-KAN model
    n_pre.load_state_dict(checkpoint['state_dict'])

    return n_pre, checkpoint['VS'], checkpoint['Diff_param']

#_#_===========================================================================
def Boundary_Cond(N, x_interval, t_interval):
    """
    Generate boundary condition points for the PDE problem.
    
    Args:
        N (int): Number of boundary points to sample for each boundary.
        x_interval (list): Spatial domain interval (unused here, assumed fixed at [-1,1]).
        t_interval (list): Temporal domain interval.
    
    Returns:
        X (array): Boundary coordinates of shape (2N, 2).
        U (array): Boundary values (zeros) of shape (2N, 1).
    """
    at = t_interval[0]  # Start of time domain
    bt = t_interval[1]  # End of time domain

    # Right boundary: x = 1
    xr = np.ones((N, 1))
    tr = (at + (bt - at) * np.random.random((N, 1)))
    X_R = np.hstack((xr, tr))
    
    # Left boundary: x = -1
    xl = -1.0 * np.ones((N, 1))
    tl = (at + (bt - at) * np.random.random((N, 1)))
    X_L = np.hstack((xl, tl))
    
    # Combine left and right boundary points
    X = np.vstack((X_R, X_L))
    
    # Shuffle the points for randomness
    ind = np.arange(2*N)
    np.random.shuffle(ind)
    X = X[ind, :]
    
    # Boundary condition values are zeros
    U = 0.0 * np.ones((2*N,1))
    
    return X, U.flatten()[:, None]

#_#_===========================================================================
def Initial_Cond(N, VS, t_interval):
    """
    Generate initial condition points for the PDE problem.
    
    Args:
        N (int): Number of initial condition points.
        VS (float): Scaling parameter (used for initial condition function).
        t_interval (list): Temporal domain interval.
    
    Returns:
        X (array): Initial condition coordinates of shape (N, 2).
        U (array): Initial condition values at time t=0.
    """
    ax = -1.0  # Left spatial boundary
    bx = 1.0   # Right spatial boundary
    
    at = t_interval[0]  # Initial time (t=0)

    # Sample spatial points uniformly in the domain
    x = (ax + (bx - ax) * np.random.random((N, 1)))
    t = at * np.ones((N, 1))  # Fixed initial time
    
    # Combine spatial and temporal points
    X = np.hstack((x, t))
    
    # Shuffle the points
    ind = np.arange(N)
    np.random.shuffle(ind)
    X = X[ind, :]
    
    # Compute initial condition values (sinusoidal profile)
    U = np.sin(np.pi * VS * X[:, 0:1])
    
    return X, U.flatten()[:, None]
        
#_#_===========================================================================
def X_dom(N, x_interval, t_interval):
    """
    Generate random interior points in the spatio-temporal domain.
    
    Args:
        N (int): Number of interior points.
        x_interval (list): Spatial domain interval (unused here, assumed fixed at [-1,1]).
        t_interval (list): Temporal domain interval.
    
    Returns:
        X (array): Interior points of shape (N, 2).
    """
    ax = -1.0  # Left spatial boundary
    bx = 1.0   # Right spatial boundary
    
    at = t_interval[0]  # Start of time domain
    bt = t_interval[1]  # End of time domain
    
    # Sample random spatial and temporal points
    x = (ax + (bx - ax) * np.random.random((N, 1)))
    t = (at + (bt - at) * np.random.random((N, 1)))
    
    X = np.hstack((x, t))
    
    return X

#_#_===========================================================================
