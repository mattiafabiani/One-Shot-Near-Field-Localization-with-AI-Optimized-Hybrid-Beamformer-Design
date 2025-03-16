"""
University of Bologna

Description: Pytorch deep neural network model definitions
Author: Mattia Fabiani
Date: 9/04/2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DNN_model(nn.Module):
    '''
    Deep Neural Network (DNN) model for localization.

    Args:
        N (int): Number of antennas.
        N_RF (int): Number of RF chains.
        n_out (int): Number of output features.

    Layers:
        fc1: Fully connected layer to transform input features.
        mlp: Multi-layer perceptron with ReLU activations and Tanh output.
    '''
    def __init__(self, N, N_RF, n_out):
        super(DNN_model, self).__init__()
        self.N = N
        self.N_RF = N_RF
        self.fc1 = nn.Linear(self.N*2, self.N_RF*2)
        self.mlp = nn.Sequential(
            nn.Linear(self.N_RF*2,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,n_out), nn.Tanh(),
        )
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.mlp(x)
        return x
    
class CNN_model(nn.Module):
    '''
    Convolutional Neural Network (CNN) model for near-field localization.

    Args:
        N (int): Number of antennas.
        N_RF (int): Number of RF chains.
        n_out (int): Number of output features.

    Layers:
        fc1: Fully connected layer to transform input features.
        conv1: First convolutional layer with BatchNorm and ReLU activation.
        conv2: Second convolutional layer with BatchNorm and ReLU activation.
        mlp: Multi-layer perceptron with BatchNorm, ReLU activations, and Dropout.
    '''
    def __init__(self, N, N_RF, n_out):
        super(CNN_model, self).__init__()
        self.N = N
        self.N_RF = N_RF
        conv1_channels = 32
        conv2_channels = 16
        self.fc1 = nn.Linear(self.N*2, self.N_RF*2)
        self.mlp = nn.Sequential(
            nn.Linear(self.N_RF*conv2_channels,512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512,256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(.1),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(.1),
            nn.Linear(128,n_out), nn.Tanh()
        )
        self.conv1 = nn.Sequential(nn.Conv2d(2,conv1_channels,1,1), nn.BatchNorm2d(conv1_channels), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(conv1_channels,conv2_channels,1,1), nn.BatchNorm2d(conv2_channels), nn.ReLU())

    def forward(self, x):
        batch_size = x.size(0)

        # Forward pass through the first layer
        x = self.fc1(x)  # Output shape: [batch_size, N_RF * 2]
        x = x.view(batch_size, self.N_RF, 2)  # Output shape: [batch_size, N_RF, 2]
        x = x.permute(0, 2, 1).unsqueeze(3)  # Output shape: [batch_size, 2, N_RF, 1]
        x = self.conv1(x)  # Output shape: [batch_size, 32, N_RF, 1]
        x = self.conv2(x)  # Output shape: [batch_size, 16, N_RF, 1]
        x = x.view(batch_size, -1)  # Output shape: [batch_size, 16 * N_RF]
        x = self.mlp(x)  # Output shape: [batch_size, n_out]
        return x
    
class weightConstraint(object):
    '''
    Weight constraint class to enforce specific structures on the weights of the neural network layers.

    Args:
        N (int): Number of antennas.
        N_RF (int): Number of RF chains.
        type (str): Type of weight constraint ('fully-connected', 'sub-connected', 'inter-connected').

    Methods:
        __call__(module): Applies the weight constraint to the given module.
    '''
    def __init__(self, N, N_RF, type='fully-connected'):
        self.N = N
        self.N_RF = N_RF
        self.type = type
        if type == 'sub-connected':
            self.n_ant_per_rf = int(np.floor(self.N / self.N_RF))
            self.mask = np.kron(np.eye(2 * self.N_RF), np.ones((1, self.n_ant_per_rf)))  # cancel everything except the block-wise diagonal (sub-connected structure)
        pass

    def __call__(self, module):
        if hasattr(module, 'weight'):
            if self.type == 'fully-connected':
                '''
                Apply fully-connected weight constraint.
                '''
                w = module.weight.data  # shape (2*N_RF, 2*N)
                w[self.N_RF:, :self.N] = 0
                w[:self.N_RF, self.N:] = 0
                w_rf = w[:self.N_RF, :self.N] + 1j * w[self.N_RF:self.N_RF * 2, self.N:self.N * 2]  # shape (N_RF, N)
                w[:self.N_RF, :self.N] /= (torch.abs(w_rf) * torch.sqrt(torch.tensor(self.N)))
                w[self.N_RF:self.N_RF * 2, self.N:self.N * 2] /= (torch.abs(w_rf) * torch.sqrt(torch.tensor(self.N)))
                w1 = w[:self.N_RF, :self.N]
                w2 = w[self.N_RF:self.N_RF * 2, self.N:self.N * 2]
                w_upper = torch.cat([w1, -w2], dim=1)
                w_lower = torch.cat([w2, w1], dim=1)
                w = torch.cat([w_upper, w_lower], dim=0)
                module.weight.data = w

            elif self.type == 'sub-connected':
                '''
                Apply sub-connected weight constraint.
                This constraint ensures that the weights are structured in a block-wise diagonal manner for sub-connected layers.
                '''
                w = module.weight.data  # shape (2*N_RF, 2*N)
                w = w * torch.tensor(self.mask, dtype=torch.float32)
                w_rf = w[:self.N_RF, :self.N] + 1j * w[self.N_RF:self.N_RF * 2, self.N:self.N * 2]  # shape (N_RF, N)
                norm_factor = torch.clamp(torch.abs(w_rf), min=1e-8) * torch.sqrt(torch.tensor(self.N / self.N_RF, dtype=torch.float32))
                w[:self.N_RF, :self.N] /= norm_factor
                w[self.N_RF:self.N_RF * 2, self.N:self.N * 2] /= norm_factor
                w1 = w[:self.N_RF, :self.N]
                w2 = w[self.N_RF:self.N_RF * 2, self.N:self.N * 2]
                w_upper = torch.cat([w1, -w2], dim=1)
                w_lower = torch.cat([w2, w1], dim=1)
                w = torch.cat([w_upper, w_lower], dim=0)
                module.weight.data = w

            elif self.type == 'inter-connected':
                '''
                Apply inter-connected weight constraint.
                This constraint ensures that the weights are structured in an interleaved manner for inter-connected layers.
                '''
                w = module.weight.data  # shape (2*N_RF, 2*N)
                indices = torch.arange(self.N) % self.N_RF  # it creates an array of inter-indeces
                interleaved_mask = torch.zeros_like(w)
                interleaved_mask[indices, torch.arange(self.N)] = 1.0
                interleaved_mask[self.N_RF + indices, self.N + torch.arange(self.N)] = 1.0
                w = w * interleaved_mask  # Apply the new mask
                w_rf = w[:self.N_RF, :self.N] + 1j * w[self.N_RF:self.N_RF * 2, self.N:self.N * 2]  # shape (N_RF, N)
                norm_factor = torch.clamp(torch.abs(w_rf), min=1e-8) * torch.sqrt(torch.tensor(self.N / self.N_RF, dtype=torch.float32))
                w[:self.N_RF, :self.N] /= norm_factor
                w[self.N_RF:self.N_RF * 2, self.N:self.N * 2] /= norm_factor
                w1 = w[:self.N_RF, :self.N]
                w2 = w[self.N_RF:self.N_RF * 2, self.N:self.N * 2]
                w_upper = torch.cat([w1, -w2], dim=1)
                w_lower = torch.cat([w2, w1], dim=1)
                w = torch.cat([w_upper, w_lower], dim=0)
                module.weight.data = w