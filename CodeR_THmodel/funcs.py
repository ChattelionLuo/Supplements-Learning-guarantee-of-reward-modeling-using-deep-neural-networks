import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as Data
from torch.autograd import Variable,grad
import random
import time
import os
torch.set_default_dtype(torch.float64)

class FNN(nn.Module):
    def __init__(self, dim_vec):
        super(FNN, self).__init__()
        layers = []
        for i in range(len(dim_vec) - 1):
            layers.append(nn.Linear(dim_vec[i], dim_vec[i+1]))
            if i < len(dim_vec) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.initialize_weights()

    def forward(self, x):
        return self.network(x)
    
    def initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)

def Hg(x: torch.Tensor) -> torch.Tensor:
    C = 2.0 / (2.0 * torch.sqrt(torch.tensor(15.0)) * torch.pow(torch.tensor(torch.pi), 0.25))
    poly = 4 * torch.pow(x, 5) - 20 * torch.pow(x, 3) + 15 * x
    gaussian = torch.exp(-0.5 * torch.pow(x, 2))
    
    return C * poly * gaussian

def Ss(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)+torch.sin(x**2)


def generate_synthetic_dataset(n, d,model,true_weights,seed):
    torch.manual_seed(seed)
    if model == 'bt': # Bradley-Terry model

        state_features = torch.rand((n, d))
        state_features2 = state_features
        for k in range(d):
            state_features2[:,k] = torch.sin(state_features[:,k])
        r1 = 2*torch.sin(4*state_features2 @ true_weights)#Hg(4*state_features2 @ true_weights)#
        r0 = -2*torch.sin(4*state_features2 @ true_weights)#Hg(4*state_features2 @ true_weights)#
        
        probabilities = torch.sigmoid(r1 - r0) #sine function

    elif model == 'thurstonian':

        state_features = torch.rand((n, d))
        state_features2 = state_features
        for k in range(d):
            state_features2[:,k] = torch.sin(state_features[:,k])
        r1 = 2*torch.sin(4*state_features2 @ true_weights)#Hg(4*state_features2 @ true_weights)#
        r0 = -2*torch.sin(4*state_features2 @ true_weights)#Hg(4*state_features2 @ true_weights)#
        normal_dist = torch.distributions.Normal(0, 1)
        probabilities = normal_dist.cdf(r1-r0) #sine function
    else:
        raise ValueError("Model must be 'bt' or 'thurstonian'")
    preferences = torch.bernoulli(probabilities)
    dataset = TensorDataset(state_features, preferences)
    return dataset

    
def generate_synthetic_dataset2(n, d, model, true_weights, seed, noise_rate):

    torch.manual_seed(seed)
    if model == 'bt': # Bradley-Terry model
        state_features = torch.rand((n, d))
        state_features2 = state_features
        for k in range(d):
            state_features2[:,k] = torch.sin(state_features[:,k])
        r1 = 2*torch.sin(4*state_features2 @ true_weights)
        r0 = -2*torch.sin(4*state_features2 @ true_weights)
        
        probabilities = torch.sigmoid(r1 - r0) #sine function

    elif model == 'thurstonian':
        state_features = torch.rand((n, d))
        state_features2 = state_features
        for k in range(d):
            state_features2[:,k] = torch.sin(state_features[:,k])
        r1 = 2*torch.sin(4*state_features2 @ true_weights)
        r0 = -2*torch.sin(4*state_features2 @ true_weights)
        normal_dist = torch.distributions.Normal(0, 1)
        probabilities = normal_dist.cdf(r1-r0) #sine function
    else:
        raise ValueError("Model must be 'bt' or 'thurstonian'")
    
    probabilities = probabilities.view(-1, 1)
    
    if noise_rate > 0:
        n_noise = int(n * noise_rate)
        noise_indices = torch.randperm(n)[:n_noise]
        noisy_probs = torch.rand(n_noise, 1) * 0.4 + 0.3
        probabilities[noise_indices] = noisy_probs
    
    preferences = torch.bernoulli(probabilities)
    
    dataset = TensorDataset(state_features, preferences)
    return dataset


def generate_filtered_dataset(n, d, model,true_weights, seed, band):

    lower_bound=0.5-band
    upper_bound=0.5+band

    torch.manual_seed(seed)
    if model == 'bt': # Bradley-Terry model

        state_features = torch.rand((5*n, d))
        state_features2 = state_features
        for k in range(d):
            state_features2[:,k] = torch.sin(state_features[:,k])
        r1 = 2*torch.sin(4*state_features2 @ true_weights)
        r0 = -2*torch.sin(4*state_features2 @ true_weights)
        
        probabilities = torch.sigmoid(r1 - r0) #sine function

    elif model == 'thurstonian':

        state_features = torch.rand((5*n, d))
        state_features2 = state_features
        for k in range(d):
            state_features2[:,k] = torch.sin(state_features[:,k])
        r1 = 2*torch.sin(4*state_features2 @ true_weights)
        r0 = -2*torch.sin(4*state_features2 @ true_weights)
        normal_dist = torch.distributions.Normal(0, 1)
        probabilities = normal_dist.cdf(r1-r0) #sine function
    else:
        raise ValueError("Model must be 'bt' or 'thurstonian'")
    
    # Filter probabilities
    mask = (probabilities < lower_bound) | (probabilities > upper_bound)
    filtered_features = state_features[mask.squeeze()]
    filtered_preferences = torch.bernoulli(probabilities[mask])
    # Randomly select exactly n samples if possible
    if len(filtered_features) >= n:
        indices = torch.randperm(len(filtered_features))[:n]
        filtered_features = filtered_features[indices]
        filtered_preferences = filtered_preferences[indices]
    else:
        print("Warning: Not enough data after filtering. Generated fewer samples.")

    dataset = TensorDataset(filtered_features, filtered_preferences.view(-1, 1))
    return dataset