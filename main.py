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
		self.e = args.e
		self.N = args.n
		if(args.load_critic):
			self.critic_model.load_state_dict(torch.load('/home/nihar/Desktop/DeepRL/Curiosity/models/N_'+str(self.N)+'/critic/'+'critic_model_'+'N_'+str(self.N)+'_episode_'+str(self.e)))
		if(args.load_actor):
			self.actor_model.load_state_dict(torch.load('/home/nihar/Desktop/DeepRL/Curiosity/models/N_'+str(self.N)+'/actor/'+'actor_model_'+'N_'+str(self.N)+'_episode_'+str(self.e)))
		self.actor_lr = args.actor_lr
		self.critic_lr = args.critic_lr
		self.env = env
		self.num_episodes = args.num_episodes

	def generate_episode(self, env):
		states = []
		actions = []
		actions_probdist = []
		rewards = []
		state = env.reset()
		e = 0
		while True:
			e += 1
			actions_arr = torch.arange(self.env.action_space.n)
			state = self.features(state)
			action_probs = self.actor_model(state)
			actions_probdist.append(action_probs)
			p = np.random.random(1)
			if(np.random.random(1) < 0.3):
				action = np.random.randint(env.action_space.n)
			else:
				action = np.random.choice(actions_arr, p=action_probs.detach().numpy())
			next_state, reward, done, info = env.step(action)
			actions.append(action)
			states.append(state)
			rewards.append(reward)
			state = next_state
			if(e>30000):
				if(sum(rewards)>0):
					print('Length of Episode: '+str(len(actions))+', Reward: '+str(sum(rewards)))
					break
				else:
					states = []
					actions = []
					rewards = []
		return torch.stack(states), torch.LongTensor(actions), torch.FloatTensor(rewards), torch.stack(actions_probdist)

	def train(self, args, gamma=1.0):
		actor_optimizer = torch.optim.Adam(self.actor_model.parameters(), lr=self.actor_lr)
		critic_optimizer = torch.optim.Adam(self.critic_model.parameters(), lr=self.critic_lr)
		e = self.e
		while e<self.num_episodes:
			e+=1
			states, actions, rewards, action_probs = self.generate_episode(self.env)
			
			T = len(states)
			N = self.N
			R = torch.FloatTensor([0]*T)
			ICM_loss = 0
			t2 = 0 #just a tmp for ICMLoss iteration
			for t in range(T-1, 0, -1):
				V_end = 0 if (t+N>=T) else self.critic_model.forward(states[t+N])[actions[t+N]]
				R[t] = (gamma**N)*V_end
				ICM_loss += self.ICMLoss(states[t2], states[t2+1], action_probs[t2])
				t2 += 1
				#pdb.set_trace()
				tmp = 0
				for k in range(N):
					tmp += (gamma**k)*rewards[t+k] if(t+k<T) else 0
				R[t] += tmp
			pi_A_S = self.actor_model(states)
			log_prob = torch.log(pi_A_S)
			log_probs = torch.zeros(len(log_prob))
			actions = actions.long()
			for i in range(len(actions)):
				log_probs[i] = log_prob[i][actions[i]]
			Vw_St, _ = torch.max(self.critic_model(states),dim=1)
			scaling_factor = (R.detach() - Vw_St.detach())
			L_theta = torch.mean(scaling_factor*(-log_probs))
			L_w = torch.mean((R - Vw_St)**2)
			actor_optimizer.zero_grad()
			critic_optimizer.zero_grad()
			tmp = L_theta + L_w + ICM_loss
			tmp.backward()
			actor_optimizer.step()
			critic_optimizer.step()
			#if(e%100==0):
			#	torch.save(self.actor_model.state_dict(),'/home/nihar/Desktop/DeepRL/Curiosity/models/N_'+str(self.N)+'/actor/'+'actor_model_'+'N_'+str(self.N)+'_episode_'+str(e))
			#	torch.save(self.critic_model.state_dict(),'/home/nihar/Desktop/DeepRL/Curiosity/models/N_'+str(self.N)+'/critic/'+'critic_model_'+'N_'+str(self.N)+'_episode_'+str(e))
			
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
	parser.add_argument('--lr', dest='actor_lr', type=float,
						default=5e-4, help="The actor's learning rate.")
	parser.add_argument('--critic-lr', dest='critic_lr', type=float,
						default=1e-4, help="The critic's learning rate.")
	parser.add_argument('--n', dest='n', type=int,
						default=100, help="The value of N in N-step A2C.")
	parser.add_argument('--load_critic', dest='load_critic', type=bool,
						default=False, help="Load the critic model")
	parser.add_argument('--load_actor', dest='load_actor', type=bool,
						default=False, help="Load the actor model")
	parser.add_argument('--load_episode', dest='e', type=int,
						default=23100, help="Load the actor model")
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
	curiosity.train(args)

if __name__ == '__main__':
	main(sys.argv)
