import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import pdb


# Policy Model

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        global num_states, num_actions

        self.ff1 = nn.Linear(num_states,8, bias=True)
        torch.nn.init.xavier_uniform_(self.ff1.weight, gain=1)
        self.relu = nn.ReLU()

        self.ff2 = nn.Linear(8,16,bias=True)
        torch.nn.init.xavier_uniform_(self.ff2.weight, gain=1)
        self.ff3 = nn.Linear(16,16,bias=True)
        torch.nn.init.xavier_uniform_(self.ff3.weight, gain=1)

        self.ff4 = nn.Linear(16,16,bias=True)
        torch.nn.init.xavier_uniform_(self.ff4.weight, gain=1)

        self.final = nn.Linear(16,num_actions,bias=True)
        torch.nn.init.xavier_uniform_(self.final.weight, gain=1)

    def forward(self, input):
        input = torch.FloatTensor(input)
        ff1 = self.relu(self.ff1(input))
        ff2 = self.relu(self.ff2(ff1))
        ff3 = self.relu(self.ff3(ff2))
        ff4 = self.relu(self.ff4(ff3))
        output = F.softmax(self.final(ff4))
        return output


# Value Model

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        global num_states, num_actions

        self.ff1 = nn.Linear(num_states,8, bias=True)
        torch.nn.init.xavier_uniform_(self.ff1.weight, gain=1)
        self.relu = nn.ReLU()

        self.ff2 = nn.Linear(8,16,bias=True)
        torch.nn.init.xavier_uniform_(self.ff2.weight, gain=1)
        self.ff3 = nn.Linear(16,16,bias=True)
        torch.nn.init.xavier_uniform_(self.ff3.weight, gain=1)

        self.ff4 = nn.Linear(16,8,bias=True)
        torch.nn.init.xavier_uniform_(self.ff4.weight, gain=1)

        self.final = nn.Linear(8,num_actions,bias=True)
        torch.nn.init.xavier_uniform_(self.final.weight, gain=1)

    def forward(self, input):
        input = torch.FloatTensor(input)
        ff1 = self.relu(self.ff1(input))
        ff2 = self.relu(self.ff2(ff1))
        ff3 = self.relu(self.ff3(ff2))
        ff4 = self.relu(self.ff4(ff3))
        output = self.final(ff4)
        return output
