import gymnasium as gym

import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


import numpy as np


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.autograd.set_detect_anomaly(True)



class CartPolePPO:

    def __init__(self):

        # self.env = gym.make("CartPole-v1")
        self.env = gym.make("CartPole-v1", render_mode="human")

        ## Parameteres
        self.n_disc_action = 2 # Since action is discrete (0 = left or 1 = right)
        self.dim_action = 1
        self.n_state = 4

        self.horizon = 500
        self.n_episodes = 60
        self.n_updates_per_ep = 10

        self.freq_eval = 2

        self.gamma = 0.99
        self.policy_lr = 0.0003
        self.vf_lr = 1e-3

        self.ppo_clip = 0.5
        self.loss_entropy_coef = 0.8
        self.loss_vf_coef = 0.15

        self.normalize_adv = True

        self.init_nn()  

        self.opt_policy = Adam(self.actor_net.parameters(), lr=self.policy_lr, eps=1e-5)

        self.opt_vf = RMSprop(self.vf_net.parameters(), lr=self.vf_lr)
        self.loss_vf_fn = torch.nn.MSELoss()

        params = list(self.actor_net.parameters()) + list(self.vf_net.parameters())
        self.opt_policy_vf = Adam(params, lr=self.policy_lr, eps=1e-5)

        self.rtg_horizon = torch.zeros(self.horizon)


    def init_weights(self,m):
        if type(m) == nn.Linear:
            m.weight.data.fill_(0.0)


    def init_nn(self):
        
        self.actor_arch = 64
        self.vf_arch = 64

        self.vf_net = nn.Sequential(
            nn.Linear(self.n_state, self.vf_arch),
            nn.Tanh(),
            nn.Linear(self.vf_arch, self.vf_arch),
            nn.Tanh(),
            nn.Linear(self.vf_arch, self.vf_arch),
            nn.Tanh(),
            nn.Linear(self.vf_arch, 1),
        )
        # self.vf_net.apply(self.init_weights)

        self.actor_net = nn.Sequential(
            nn.Linear(self.n_state, self.actor_arch),
            nn.Tanh(),
            nn.Linear(self.actor_arch, self.actor_arch),
            nn.Tanh(),
            nn.Linear(self.actor_arch, self.actor_arch),
            nn.Tanh(),
            nn.Linear(self.actor_arch, self.n_disc_action),
        )
        # self.actor_net.apply(self.init_weights)



    def build_dataset_from_policy(self):

        obs = self.env.reset()
        obs = obs[0]

        self.obs_horizon = torch.zeros([self.horizon, self.n_state])
        self.new_obs_horizon = torch.zeros([self.horizon, self.n_state])
        self.rew_horizon = torch.zeros(self.horizon)
        self.act_horizon = torch.zeros(self.horizon)
        self.old_act_horizon = torch.zeros(self.horizon)
        self.end_horizon = torch.zeros(self.horizon)
        self.log_prob_horizon = torch.zeros(self.horizon)
        self.old_log_prob_horizon = torch.zeros(self.horizon)


        for t in range(self.horizon):
            state = torch.from_numpy(obs)

            # Get Action
            action_dist = Categorical(logits=self.actor_net(state))
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(axis=-1)
            action_np = action.numpy()
            ###

            new_obs, reward, done, info, _ = self.env.step(action_np)

            self.obs_horizon[t][:] = torch.from_numpy(obs)
            self.new_obs_horizon[t][:] = torch.from_numpy(new_obs)
            self.rew_horizon[t] = reward
            self.act_horizon[t] = action
            self.end_horizon[t] = done
            self.log_prob_horizon[t] = log_prob.detach()

            # If Done, reset environment
            if done:
                # print(done)
                obs = self.env.reset()
                obs = obs[0]
            else:
                obs = new_obs

    
    def get_action(self, obs):
        action_probs = self.actor_net(torch.tensor(obs))
        return torch.argmax(action_probs)
    

    def computeRTG(self):
        for i in reversed(range(self.horizon)):
            
            if i == self.horizon-1 or self.end_horizon[i]:
                self.rtg_horizon[i] = self.rew_horizon[i]
            
            else:
                self.rtg_horizon[i] = self.rew_horizon[i] + self.gamma * self.rtg_horizon[i+1]


    def learn(self):
        obs = self.env.reset(seed=1)
        state = torch.from_numpy(obs[0])

        ## Training
        self.loss_train = []
        self.rtg_train = []
        self.vf_train = []

        for ep in range(self.n_episodes):

            self.build_dataset_from_policy()

            # adv_batch = self.computeGAEReturn()
            self.computeRTG()
            vf_value = self.vf_net(self.obs_horizon).squeeze()
            adv_batch = self.rtg_horizon - vf_value.detach()

            if self.normalize_adv:
                adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-10)

            log_prob_old_pi = self.log_prob_horizon

            for b in range(self.n_updates_per_ep):
                
                action_dist = Categorical(logits=self.actor_net(self.obs_horizon))
                log_prob_pi = action_dist.log_prob(self.act_horizon)

                ratio_pi = torch.exp(log_prob_pi - log_prob_old_pi)

                ###
                lclip_p1 = adv_batch * ratio_pi
                lclip_p2 = adv_batch * torch.clamp(ratio_pi, 1 - self.ppo_clip, 1 + self.ppo_clip)
                lclip = -torch.min(lclip_p1,lclip_p2).mean()

                loss_policy = lclip        

                ## Opt Policy
                self.opt_policy.zero_grad()
                loss_policy.backward()
                self.opt_policy.step()

            loss_updates = []
            for b in range(self.n_updates_per_ep):
                self.opt_vf.zero_grad()

                vf_value = self.vf_net(self.obs_horizon).squeeze()

                loss_vf = self.loss_vf_fn(vf_value, self.rtg_horizon)
                
                loss_vf.backward()
                self.opt_vf.step()

                loss_updates.append(loss_vf.detach().numpy())
            
            self.loss_train.append(np.mean(np.array(loss_updates)))
            self.rtg_train.append(np.mean(self.rtg_horizon.numpy()))
            self.vf_train.append(np.mean(vf_value.detach().numpy()))

            if ep % self.freq_eval == 0:
                print(f'Loss ep {ep}: {self.loss_train[-1]:.3f} | Avg RTG: {self.rtg_train[-1]:.3f}')



    def learnSingleOpt(self):
        obs = self.env.reset(seed=1)
        state = torch.from_numpy(obs[0])


        ## Training
        self.loss_train = []
        self.rtg_train = []
        self.vf_train = []

        for ep in range(self.n_episodes):

            self.build_dataset_from_policy()
            self.computeRTG()

            log_prob_old_pi = self.log_prob_horizon

            loss_updates = []
            for b in range(self.n_updates_per_ep):
                
                vf_value = self.vf_net(self.obs_horizon).squeeze()
                adv_batch = self.rtg_horizon - vf_value

                if self.normalize_adv:
                    adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-10)


                action_dist = Categorical(logits=self.actor_net(self.obs_horizon))
                log_prob_pi = action_dist.log_prob(self.act_horizon)

                ratio_pi = torch.exp(log_prob_pi - log_prob_old_pi)

                ###
                lclip_p1 = adv_batch * ratio_pi
                lclip_p2 = adv_batch * torch.clamp(ratio_pi, 1 - self.ppo_clip, 1 + self.ppo_clip)
                lclip = -torch.min(lclip_p1,lclip_p2).mean()

                lentropy = -torch.mean(-adv_batch)

                ###

                loss_vf = self.loss_vf_fn(vf_value, self.rtg_horizon)
                

                ##
                loss_policy_vf = lclip + self.loss_vf_coef * loss_vf + self.loss_entropy_coef * lentropy       

                ## Opt Policy
                self.opt_policy_vf.zero_grad()
                loss_policy_vf.backward()
                self.opt_policy_vf.step()

                loss_updates.append(loss_vf.detach().numpy())
            
            self.loss_train.append(np.mean(np.array(loss_updates)))
            self.rtg_train.append(np.mean(self.rtg_horizon.numpy()))
            self.vf_train.append(np.mean(vf_value.detach().numpy()))

            if ep % self.freq_eval == 0:
                print(f'Loss ep {ep}: {self.loss_train[-1]:.3f} | Avg RTG: {self.rtg_train[-1]:.3f}')


        print(f'Final Training Loss: {self.loss_train[-1]:.3f} | Avg RTG: {self.rtg_train[-1]:.3f}')

cart_pole_ppo = CartPolePPO()
# cart_pole_ppo.learn()
cart_pole_ppo.learnSingleOpt()


obs = cart_pole_ppo.env.reset()
obs = obs[0]
while True:
    action = cart_pole_ppo.get_action(obs)
    obs, reward, done, info, _ = cart_pole_ppo.env.step(action.numpy())
    cart_pole_ppo.env.render()
    # VecEnv resets automatically
    if done:
      obs = cart_pole_ppo.env.reset()
      obs = obs[0]