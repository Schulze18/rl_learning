from typing import Dict, List, Tuple, Type, Union

import gymnasium as gym
from gymnasium import spaces

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

class PPO:

    def __init__(self, env, n_episodes = 1000, batch_size = 500, n_updates_per_ep = 10, 
                 freq_print = 10, policy_arch = [64, 64], vf_arch = [64, 64],
                 normalize_adv = True, gamma = 0.99, policy_lr = 0.0003, vf_lr = 1e-3, 
                 use_GAE = True, lambda_gae = 0.95, target_kl = 0.01, log_std_init = -0.5,
                 ppo_clip = 0.5, loss_vf_coef = 0.5, loss_entropy_coef = 0.0, adam_eps = 1e-5):

        torch.set_default_dtype(torch.float64)

        self.env = env
        self.n_state = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, spaces.Discrete):
            self.discrete_action = True
            self.n_disc_action = self.env.action_space.n
        else:
            self.discrete_action = False
            self.n_action = self.env.action_space.shape[0]

        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.n_updates_per_ep = n_updates_per_ep

        self.freq_print = freq_print

        self.policy_arch = policy_arch
        self.vf_arch = vf_arch

        self.gamma = gamma
        self.policy_lr = policy_lr
        self.vf_lr = vf_lr
        self.ppo_clip = ppo_clip
        self.loss_vf_coef = loss_vf_coef
        self.loss_entropy_coef = loss_entropy_coef
        self.adam_eps = adam_eps

        self.lambda_gae = lambda_gae
        self.use_GAE = use_GAE
        self.target_kl = target_kl
        self.normalize_adv = normalize_adv
        self.log_std_init = log_std_init

        self.policy_net: List[nn.Module] = []
        self.vf_net: List[nn.Module] = []

        self.build_nn()

    def build_nn(self):
        policy_net: List[nn.Module] = []
        vf_net: List[nn.Module] = []

        activation_fn = nn.Tanh()

        last_layer_dim = self.n_state
        for layer_dim in self.policy_arch:
            policy_net.append(nn.Linear(last_layer_dim, layer_dim))
            policy_net.append(activation_fn)
            last_layer_dim = layer_dim
        policy_net.append(nn.Linear(last_layer_dim, last_layer_dim))
        policy_net.append(activation_fn)
        if self.discrete_action:
            policy_net.append(nn.Linear(last_layer_dim, self.n_disc_action))
        else:
            policy_net.append(nn.Linear(last_layer_dim, self.n_action))
        last_layer_dim = self.n_state
        for layer_dim in self.vf_arch:
            vf_net.append(nn.Linear(last_layer_dim, layer_dim))
            vf_net.append(activation_fn)
            last_layer_dim = layer_dim
        vf_net.append(nn.Linear(last_layer_dim, last_layer_dim))
        vf_net.append(activation_fn)
        vf_net.append(nn.Linear(last_layer_dim, 1))


        self.policy_net = nn.Sequential(*policy_net)
        self.vf_net = nn.Sequential(*vf_net)

        log_std = self.log_std_init * np.ones(self.n_action, dtype=np.float64)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

        self.loss_vf_fn = torch.nn.MSELoss()

        if self.discrete_action:
            params = {'params': self.policy_net.parameters(), 'lr': self.policy_lr}, {'params': self.vf_net.parameters(), 'lr': self.vf_lr}
        else:
            params = {'params': self.policy_net.parameters(), 'lr': self.policy_lr}, {'params': self.vf_net.parameters(), 'lr': self.vf_lr}, {'params': self.log_std, 'lr': self.policy_lr}
        
        self.opt_policy_vf = Adam(params, lr=self.policy_lr, eps=self.adam_eps)


    def buildDataset(self):

        obs = self.env.reset()
        obs = obs[0]
        
        dataset_obs = []
        dataset_act = []
        dataset_rew = []
        dataset_log_prob = []
        dataset_end = []
        dataset_new_obs = []

        for t in range(self.batch_size):
            state = torch.from_numpy(obs)

            # Get Action
            action_dist = self.get_action_dist(state)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(axis=-1)
            action_np = action.numpy()

            new_obs, reward, done, info, _ = self.env.step(action_np)

            dataset_obs.append(torch.from_numpy(obs))
            dataset_new_obs.append(torch.from_numpy(new_obs))
            dataset_act.append(action)
            dataset_rew.append(reward)
            dataset_log_prob.append(log_prob)
            
            if done:
                obs = self.env.reset()
                obs = obs[0]
            else:
                obs = new_obs


            dataset_end.append(done)

        dataset_obs = torch.tensor(np.array(dataset_obs), dtype=torch.float64)
        dataset_new_obs = torch.tensor(np.array(dataset_new_obs), dtype=torch.float64)
        dataset_act = torch.tensor(np.array(dataset_act), dtype=torch.float64)
        dataset_rtg = self.computeRTG(dataset_rew, dataset_end, dataset_obs)
        dataset_log_prob = torch.tensor(dataset_log_prob, dtype=torch.float64)
        dataset_end = torch.tensor(dataset_end, dtype=torch.float64)
        dataset_rew = torch.tensor(np.array(dataset_rew), dtype=torch.float64)

        return dataset_obs, dataset_act, dataset_rtg, dataset_end, dataset_log_prob, dataset_new_obs, dataset_rew



    def get_action(self, obs):
        if self.discrete_action:
            action_probs = self.policy_net(torch.tensor(obs))
            action = torch.argmax(action_probs)
        else:
            action = self.policy_net(torch.tensor(obs))
        return action.detach()
    
    
    def get_action_dist(self, obs):
        if self.discrete_action:
            action_dist = Categorical(logits=self.policy_net(obs))
        else:
            mean_action = self.policy_net(obs)
            std_action = torch.exp(self.log_std)
            action_dist = Normal(mean_action, std_action)

        return action_dist
    

    def computeRTG(self, batch_rew, batch_end, batch_obs):
        
        dataset_rtg = []

        for i in reversed(range(len(batch_rew))):
            
            if batch_end[i]:
                dataset_rtg.insert(0, batch_rew[i])

            elif i == len(batch_rew)-1:
                dataset_rtg.insert(0, self.vf_net(batch_obs[i]).detach().numpy()[0])
            
            else:
                dataset_rtg.insert(0, batch_rew[i] + self.gamma * dataset_rtg[0])

        dataset_rtg = torch.tensor(dataset_rtg, dtype=torch.float64)
        return dataset_rtg
    

    def computeGAE(self, dataset_obs, dataset_new_obs, dataset_rew, dataset_end):

        adv = []

        vf = self.vf_net(dataset_obs).squeeze()
        vf_next = self.vf_net(dataset_new_obs).squeeze()

        with torch.no_grad():
            for i in reversed(range(len(dataset_obs))):
            
                if dataset_end[i]:
                    adv.insert(0, dataset_rew[i] - vf[i])

                elif i == len(dataset_rew)-1:
                    adv.insert(0, dataset_rew[i] + self.gamma * vf_next[i] - vf[i])

                else:
                    adv.insert(0, dataset_rew[i] + self.gamma * vf_next[i] - vf[i] + self.gamma * self.lambda_gae * adv[0])

        adv = torch.tensor(adv, dtype=torch.float64)
        return vf, adv


    def learn(self):

        self.loss_train = []
        self.rtg_train = []
        self.vf_train = []
        self.approx_kl_train = []
        self.adv_train = []
        self.loss_train = []
        self.loss_entropy_train = []
        self.loss_vf_train = []
        self.lclip_train = []

        for ep in range(self.n_episodes):
            dataset_obs, dataset_act, dataset_rtg, dataset_end, dataset_log_prob, dataset_new_obs, dataset_rew = self.buildDataset() 

            old_log_prob_pi = dataset_log_prob

            if ep == 0:
                dataset_first_obs = dataset_obs

            approx_kl_updates = []
            adv_updates = []
            loss_updates = []
            loss_entropy_updates = []
            loss_vf_updates = []
            lclip_updates = []

            for it in range(self.n_updates_per_ep):

                if self.use_GAE:
                    _, adv = self.computeGAE(dataset_obs, dataset_new_obs, dataset_rew, dataset_end)
                    vf_value = self.vf_net(dataset_obs).squeeze()
                    vf_compare = dataset_rtg
                else:
                    vf_value = self.vf_net(dataset_obs).squeeze()
                    adv = dataset_rtg - vf_value.detach()
                    vf_compare = dataset_rtg

                if self.normalize_adv:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-10)

                action_dist = self.get_action_dist(dataset_obs)
                log_prob_pi = action_dist.log_prob(dataset_act).sum(axis=-1)
                log_ratio = log_prob_pi - old_log_prob_pi

                ratio_pi = torch.exp(log_ratio)

                ###
                lclip_p1 = adv * ratio_pi
                lclip_p2 = adv * torch.clamp(ratio_pi, 1 - self.ppo_clip, 1 + self.ppo_clip)
                lclip = -torch.min(lclip_p1,lclip_p2).mean()

                lentropy = -torch.mean(-adv)

                ###
                loss_vf = self.loss_vf_fn(vf_value, vf_compare)

                ##
                loss_policy_vf = lclip + self.loss_vf_coef * loss_vf + self.loss_entropy_coef * lentropy       


                self.opt_policy_vf.zero_grad()
                loss_policy_vf.backward()
                self.opt_policy_vf.step()

                loss_updates.append(loss_policy_vf.detach().numpy())
                adv_updates.append(adv.detach().numpy())
                lclip_updates.append(lclip.detach().numpy())
                loss_vf_updates.append(loss_vf.detach().numpy())
                loss_entropy_updates.append(lentropy.detach().numpy())

                with torch.no_grad():

                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_updates.append(approx_kl)
                
                if approx_kl > 1.5 * self.target_kl:
                    # print(f'Approx KL {approx_kl}, early stop')
                    break
            

            self.adv_train.append(np.mean(np.array(adv_updates)))
            
            self.loss_train.append(np.mean(np.array(loss_updates)))
            self.lclip_train.append(np.mean(np.array(lclip_updates)))
            self.loss_vf_train.append(np.mean(np.array(loss_vf_updates)))
            self.loss_entropy_train.append(np.mean(np.array(loss_entropy_updates)))

            self.rtg_train.append(np.mean(dataset_rtg.numpy()))
            self.vf_train.append(np.mean(vf_value.detach().numpy()))
            self.approx_kl_train.append(approx_kl_updates)
            self.adv_train.append(adv_updates)

            if ep % self.freq_print == 0:
                print(f'Loss ep {ep}: {self.loss_train[-1]:.3f} | Lclip: {self.lclip_train[-1]:.3f} | Lvf: {self.loss_vf_train[-1]:.3f} | Lent: {self.loss_entropy_train[-1]:.3f}  | Avg RTG: {self.rtg_train[-1]:.3f}')
        
        print(f'Final Training Loss: {self.loss_train[-1]:.3f} | Lclip: {self.lclip_train[-1]:.3f} | Lvf: {self.loss_vf_train[-1]:.3f} | Lent: {self.loss_entropy_train[-1]:.3f}  | Avg RTG: {self.rtg_train[-1]:.3f}')
