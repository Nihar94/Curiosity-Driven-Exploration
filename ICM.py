import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import gym
import pdb

torch.manual_seed(10)
np.random.seed(10)

def weights_init(m):
	classname = m.__class__.__name__
	if(classname=='Linear'):
		torch.nn.init.xavier_uniform_(m.weight)

class ICMFeatures(nn.Module):
	def __init__(self):
		super(ICMFeatures, self).__init__()
		# Fusion multiplier for Visual Features
		self.alpha = Variable(torch.randn(1), requires_grad=True)*0+1
		
		# Fusion multiplier for Scent
		self.beta = Variable(torch.randn(1), requires_grad=True)*0+1
		
		# Learnable classifier1
		self.vision_features = nn.Sequential(
			nn.Linear(11*11, 50),
			nn.LeakyReLU(inplace=True),
			nn.Linear(50, 10),
			nn.LeakyReLU(inplace=True)
			)
		self.vision_features.apply(weights_init)
		# Learnable classifier2
		self.combined_features = nn.Sequential(
			nn.Linear(28, 28),
			nn.LeakyReLU(inplace=True),
			nn.Linear(28, 10),
			nn.LeakyReLU(inplace=True)
			)
		self.combined_features.apply(weights_init)

	def forward(self, states):
		prev_scent = torch.from_numpy(states[0]['scent'])
		curr_scent = torch.from_numpy(states[1]['scent'])
		
		prev_vision = torch.from_numpy(states[0]['vision']).permute(2,0,1).unsqueeze(0)
		curr_vision = torch.from_numpy(states[1]['vision']).permute(2,0,1).unsqueeze(0)
		
		prev_moved = int(states[0]['moved'] == True)*10
		curr_moved = int(states[1]['moved'] == True)*10
		
		vision_features = torch.cat((prev_vision, curr_vision), 0)
		
		vision_features = self.cnns(vision_features)
		vision_features = vision_features.view(vision_features.size(0), 3*2*2)
		
		vision_features = self.alpha * self.vision_features(vision_features).view(2, -1)
		
		scent = torch.cat((prev_scent, curr_scent), 0)
		scent = self.beta * scent
		movement = torch.tensor([prev_moved, curr_moved]).float()
		movement.requires_grad=True
		
		combined_features = torch.cat((vision_features, scent, movement), 0)
		combined_features = self.combined_features(combined_features)
		return combined_features


class ForwardModel(nn.Module):
	def __init__(self):
		super(ForwardModel, self).__init__()
		self.network = nn.Sequential(
			nn.Linear(13, 10),
			nn.LeakyReLU(inplace=True),
			nn.Linear(10,10)
			)
		self.network.apply(weights_init)
		self.sm = nn.Softmax(dim=0)

	def forward(self, state_features, action_prob):
		representation = torch.cat((state_features, action_prob), 0)
		next_state_features = self.network(representation)
		return next_state_features


class InverseModel(nn.Module):
	def __init__(self):
		super(InverseModel, self).__init__()
		self.network = nn.Sequential(
			nn.Linear(20, 10),
			nn.LeakyReLU(inplace=True),
			nn.Linear(10,3)
			)
		self.network.apply(weights_init)
		self.sm = nn.Softmax(dim=0)

	def forward(self, state_features, next_state_features):
		a_cap = self.network(state_features, next_state_features)
		return a_cap

	def inv_loss(self, pred_action_prob, action_prob):
		L1 = torch.abs(predicted_action_prob - action_prob)
		L1 = L1.mean()
		return L1

class IntrinsicReward():
	#def __init__(self):

	def distance(self, predicted_next_state_features, next_state_features, n = 0.3):
		Lf = (1/2)*(predicted_next_state_features - next_state_features)**2
		Lf = Lf.mean()
		ri = (n/2)*(predicted_next_state_features - next_state_features)**2
		return Lf, ri