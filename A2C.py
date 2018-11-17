import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ICM import *
from A2C import *
import nel
import gym
import pdb


class Features(nn.Module):
    def __init__(self):
        super(Features, self).__init__()
        # Fusion multiplier for Visual Features
        self.alpha = Variable(torch.randn(1), requires_grad=True)*0+1
        
        # Fusion multiplier for Scent
        self.beta = Variable(torch.randn(1), requires_grad=True)*0+1
        
        # Learnable classifier1
        self.vision_features = nn.Sequential(
            nn.Linear(11*11*3, 50),
            nn.LeakyReLU(inplace=True),
            nn.Linear(50, 10),
            nn.LeakyReLU(inplace=True)
            )
        self.vision_features.apply(weights_init)
        # Learnable classifier2
        self.combined_features = nn.Sequential(
            nn.Linear(14, 14),
            nn.LeakyReLU(inplace=True)
            )
        self.combined_features.apply(weights_init)

    def forward(self, state):
        scent = torch.from_numpy(state['scent'])
        vision = torch.from_numpy(state['vision']).view(-1)
        moved = int(state['moved'] == True)
        vision_features = self.alpha * self.vision_features(vision)
        scent = self.beta * scent
        movement = torch.tensor([moved]).float()
        movement.requires_grad=True
        combined_features = torch.cat((vision_features, scent, movement), 0)
        combined_features = self.combined_features(combined_features)
        return combined_features

# Policy Model

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        global num_states, num_actions

        # Fusion multiplier for Visual Features
        self.alpha = Variable(torch.randn(1), requires_grad=True)*0+1
        
        # Fusion multiplier for Scent
        self.beta = Variable(torch.randn(1), requires_grad=True)*0+1
        
        # Learnable classifier1
        self.vision_features = nn.Sequential(
            nn.Linear(11*11*3, 50),
            nn.LeakyReLU(inplace=True),
            nn.Linear(50, 10),
            nn.LeakyReLU(inplace=True)
            )
        self.vision_features.apply(weights_init)
        # Learnable classifier2
        self.network = nn.Sequential(
            nn.Linear(14, 10),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10, 3),
            nn.LeakyReLU(inplace=True)
            )
        self.network.apply(weights_init)

    def forward(self, state):
        # scent = torch.from_numpy(state['scent'])
        # vision = torch.from_numpy(state['vision']).view(-1)
        # moved = int(state['moved'] == True)
        # vision_features = self.alpha * state#self.vision_features(vision)
        # scent = self.beta * scent
        # movement = torch.tensor([moved]).float()
        # movement.requires_grad=True
        # combined_features = torch.cat((vision_features, scent, movement), 0)
        actions = self.network(state)#combined_features)
        output = F.softmax(actions)
        return output


# Value Model

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        # Fusion multiplier for Visual Features
        self.alpha = Variable(torch.randn(1), requires_grad=True)*0+1
        
        # Fusion multiplier for Scent
        self.beta = Variable(torch.randn(1), requires_grad=True)*0+1
        
        # Learnable classifier1
        self.vision_features = nn.Sequential(
            nn.Linear(11*11*3, 50),
            nn.LeakyReLU(inplace=True),
            nn.Linear(50, 10),
            nn.LeakyReLU(inplace=True)
            )
        self.vision_features.apply(weights_init)
        # Learnable classifier2
        self.network = nn.Sequential(
            nn.Linear(14, 10),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10, 3),
            nn.LeakyReLU(inplace=True)
            )
        self.network.apply(weights_init)

    def forward(self, state):
        # scent = torch.from_numpy(state['scent'])
        # vision = torch.from_numpy(state['vision']).view(-1)
        # moved = int(state['moved'] == True)
        # vision_features = self.alpha * self.vision_features(vision)
        # scent = self.beta * scent
        # movement = torch.tensor([moved]).float()
        # movement.requires_grad=True
        # combined_features = torch.cat((vision_features, scent, movement), 0)
        actions = self.network(state)#combined_features)
        return actions
