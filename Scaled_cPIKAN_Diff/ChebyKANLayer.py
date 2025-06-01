"""Scaled-cPIKANs: Spatial Variable and Residual Scaling in Chebyshev-based
   Physics-informed Kolmogorov-Arnold Networks

DEVELOPED AT:
                    Department of Chemical Engineering
                    University of Utah, Salt Lake City,
                    Utah 84112, USA
                    DIRECTOR: Prof.  Salah A Faroughi

DEVELOPED BY:
                    FARINAZ MOSTAJERAN
                  
Inspired by:
SS, S., K. AR, A. KP, et al. (2024). Chebyshev polynomial-based kolmogorov-arnold networks: 
    An efficient architecture for nonlinear function approximation. 
    arXiv preprint arXiv:2405.07200                    

This implementation is inspired by Kolmogorov-Arnold Networks (KANs), replacing spline 
coefficients with Chebyshev polynomials for improved stability and approximation efficiency.

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

class ChebyKANLayer_2(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer_2, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        # Learnable Chebyshev coefficients: shape (input_dim, output_dim, degree + 1)
        self.cheby_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        
        # Xavier initialization for better convergence
        nn.init.xavier_normal_(self.cheby_coeffs)

        # Precompute indices for polynomial degrees
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))    

    def forward(self, x):
        # Ensure input is 2D (batch_size, input_dim)
        x = torch.reshape(x, (-1, self.inputdim))  

        # Normalize input to the Chebyshev domain [-1, 1] using tanh
        x = torch.tanh(x)

        # Initialize Chebyshev polynomial tensor: shape (batch_size, input_dim, degree + 1)
        cheby = torch.ones(x.shape[0], self.inputdim, self.degree + 1, device=x.device, dtype=torch.float32)
        
        # Set first-degree Chebyshev polynomials: T_1(x) = x
        if self.degree > 0:
            cheby[:, :, 1] = x
        
        # Recursively compute higher-degree Chebyshev polynomials using the recurrence relation:
        # T_n(x) = 2 * x * T_{n-1}(x) - T_{n-2}(x)
        for i in range(2, self.degree + 1):
            cheby[:, :, i] = 2 * x * cheby[:, :, i - 1].clone() - cheby[:, :, i - 2].clone()

        # Perform weighted sum of Chebyshev polynomials with learnable coefficients
        # Result shape: (batch_size, output_dim)
        y = torch.einsum('bid,iod->bo', cheby, self.cheby_coeffs)  
        y = y.view(-1, self.outdim)
        return y
