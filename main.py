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
import time

env = gym.make('NEL-v0')

class Curiosity():
	def __init__(self, args):
		global env
		self.actor_model = Actor()
		self.critic_model = Critic()
		self.features = Features()
		self.icm_features = ICMFeatures()
		self.icm_forward = ForwardModel()
		self.intrinsic_reward = IntrinsicReward()
		self.inv_model = InverseModel()
		self.N = args.n
		self.epsilon = 0.1
		if(args.load_models):
			self.actor_model.load_state_dict(torch.load('./models/actor_model'))
			self.critic_model.load_state_dict(torch.load('./models/critic_model'))
			self.features.load_state_dict(torch.load('./models/A2C_features'))
			self.icm_features.load_state_dict(torch.load('./models/icm_features'))
			self.inv_model.load_state_dict(torch.load('./models/inv_model'))
			self.icm_forward.load_state_dict(torch.load('./models/icm_forward'))

		self.actor_lr = args.actor_lr
		self.critic_lr = args.critic_lr
		self.icm_lr = args.icm_lr
		self.env = env
		self.num_episodes = args.num_episodes
		self.mem_state = env.reset()

	def play(self, args):
		env = gym.make('NEL-render-v0')
		state = env.reset()
		e = 0
		while e<500:
			env.render()
			e += 1
			state = self.features(state)
			action_probs = self.actor_model(state)
			action = np.random.randint(env.action_space.n)
			state, reward, done, info = env.step(action)

	# Generating all episodes on CPU. Loading and Unloading single states is expensive.
	def generate_episode(self):
		states = []
		actions = []
		actions_probdist = []
		rewards = []
		state = self.mem_state
		e = 0
		self.actor_model = self.actor_model
		t1 = time.time()
		while True:
			e += 1
			actions_arr = torch.arange(self.env.action_space.n)
			state = self.features(state)
			action_probs = self.actor_model(state)
			actions_probdist.append(action_probs)
			p = np.random.random(1)
			if(np.random.random(1) < self.epsilon):
				action = np.random.randint(env.action_space.n)
			else:
				action = np.random.choice(actions_arr, p=action_probs.detach().numpy())
			next_state, reward, done, info = env.step(action)
			actions.append(action)
			states.append(state)
			rewards.append(reward)
			state = next_state
			if(reward>0):
				t2 = time.time()
				self.mem_state = state
				print('Time taken for generating this episode: ' + str(int(t2-t1)) + ' seconds')
				break
		return torch.stack(states), torch.LongTensor(actions), torch.FloatTensor(rewards), torch.stack(actions_probdist)

	def train(self, args, gamma=1.0):
		actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=self.actor_lr)
		critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=self.critic_lr)
		#list(self.actor_model.parameters()) + list(self.critic_model.parameters()) +
		params = list(self.features.parameters()) + list(self.icm_forward.parameters()) + list(self.inv_model.parameters()) + list(self.icm_features.parameters())
		icm_optimizer = torch.optim.Adam(params, lr=args.icm_lr)
		e = 0
		mean_rewards = []
		states = []
		actions = []
		actions_probdist = []
		rewards = []
		state = env.reset()
		for i in range(100000):
			print('Step: '+str(i))
			e+=1
			if(i%100):
				self.epsilon *= 0.95
			actions_arr = torch.arange(self.env.action_space.n)
			state = self.features(state)
			action_probs = self.actor_model(state)
			actions_probdist.append(action_probs)
			p = np.random.random(1)
			if(np.random.random(1) < self.epsilon):
				action = np.random.randint(env.action_space.n)
			else:
				action = np.random.choice(actions_arr, p=action_probs.detach().numpy())
			next_state, reward, done, info = env.step(action)

			actions.append(action)
			states.append(state)
			rewards.append(reward)
			state = next_state
			np.save('rewards_a2c.npy', np.asarray(rewards))

			states1 = torch.stack(states)
			actions1 = torch.LongTensor(actions)
			rewards1 = torch.FloatTensor(rewards)
			action_probs1 = torch.stack(actions_probdist)

			self.actor_model = self.actor_model
			T = len(states)
			N = self.N
			R = torch.FloatTensor([0]*T)
			if not torch.mean(rewards1).item() == 0:
				loss_multiplier = (1/(torch.mean(rewards1).item())) # To force large updates on episodes that get rewards late and vice-versa
			mean_rewards.append(torch.mean(rewards1).item())
			print('Number of actions in episode: '+str(T))
			print('Mean Reward: '+str(torch.mean(rewards1).item()))
			print('')
			ICM_loss = 0
			t2 = 0 #just a tmp for ICMLoss iteration
			for t in range(T-1, 0, -1):
				V_end = 0 if (t+N>=T) else self.critic_model.forward(states1[t+N])[actions1[t+N]]
				R[t] = (gamma**N)*V_end
				ICM_loss += self.ICMLoss(states1[t2], states1[t2+1], action_probs1[t2])
				t2 += 1
				tmp = 0
				for k in range(N):
					tmp += (gamma**k)*rewards1[t+k] if(t+k<T) else 0
				R[t] += tmp
			pi_A_S = self.actor_model(states1)
			log_prob = torch.log(pi_A_S)
			log_probs = torch.zeros(len(log_prob))
			actions1 = actions1.long()
			for i in range(len(actions1)):
				log_probs[i] = log_prob[i][actions1[i]]
			Vw_St, _ = torch.max(self.critic_model(states1),dim=1)
			scaling_factor = (R.detach() - Vw_St.detach())

			L_theta = torch.mean(scaling_factor*(-log_probs))
			L_w = torch.mean((R - Vw_St)**2)
			actor_optimizer.zero_grad()
			critic_optimizer.zero_grad()
			icm_optimizer.zero_grad()
			if not torch.mean(rewards1).item() == 0:
				tmp = loss_multiplier*(L_theta + L_w)
			else:
				tmp = (L_theta + L_w)
			tmp.backward(retain_graph=True)
			actor_optimizer.step()
			critic_optimizer.step()
			icm_optimizer.step()
			np.save('mean_rewards_graph', mean_rewards)
			if(e%10==0):
				torch.save(self.inv_model.state_dict(),'./models/inv_model')
				torch.save(self.icm_forward.state_dict(),'./models/icm_forward')
				torch.save(self.icm_features.state_dict(),'./models/icm_features')
				torch.save(self.features.state_dict(),'./models/A2C_features')
				torch.save(self.actor_model.state_dict(),'./models/actor_model')
				torch.save(self.critic_model.state_dict(),'./models/critic_model')
		return

	def ICMLoss(self, state, next_state, action_prob, lamda = 0.5, beta = 0.3):
		pred_next_state_features = self.icm_forward(state, action_prob)
		Lf, ri = self.intrinsic_reward.distance(pred_next_state_features, next_state)
		pred_action_prob = self.inv_model(state, next_state)
		Li = self.inv_model.inv_loss(pred_action_prob, action_prob)
		loss = -lamda*(ri) + (1-beta)*Li + beta*Lf
		loss = torch.mean(loss)
		return loss

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model-config-path', dest='model_config_path',
						type=str, default='LunarLander-v2-config.json',
						help="Path to the actor model config file.")
	parser.add_argument('--num-episodes', dest='num_episodes', type=int,
						default=50000, help="Number of episodes to train on.")
	parser.add_argument('--icm-lr', dest='icm_lr', type=float,
						default=5e-4, help="The actor's learning rate.")
	parser.add_argument('--actor-lr', dest='actor_lr', type=float,
						default=5e-4, help="The actor's learning rate.")
	parser.add_argument('--critic-lr', dest='critic_lr', type=float,
						default=1e-4, help="The critic's learning rate.")
	parser.add_argument('--n', dest='n', type=int,
						default=100, help="The value of N in N-step A2C.")
	parser.add_argument('--load_models', dest='load_models', type=bool,
						default=True, help="Load all models")
	parser.add_argument('--play', dest='play', type=int,
						default=0, help="Load all models")
	parser_group = parser.add_mutually_exclusive_group(required=False)
	parser_group.add_argument('--render', dest='render',
								action='store_true',
								help="Whether to render the environment.")
	parser_group.add_argument('--no-render', dest='render',
								action='store_false',
								help="Whether to render the environment.")
	parser.set_defaults(render=False)

	return parser.parse_args()

def main(args):
	global env
	args = parse_arguments()
	curiosity = Curiosity(args)
	if(args.play==1):
		curiosity.play(args)
	else:
		curiosity.train(args)
if __name__ == '__main__':
	main(sys.argv)
