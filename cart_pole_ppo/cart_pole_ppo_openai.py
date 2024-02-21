import gymnasium as gym

import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
from torch.distributions.categorical import Categorical

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
        self.n_episodes = 20
        self.n_batch = 2
        self.n_updates_per_ep = 10
        self.size_batch = int(self.horizon / self.n_batch)

        self.freq_eval = 2

        self.gae_lambda = 0.95
        self.gamma = 0.99
        self.gae_gamma = self.gamma
        self.policy_lr = 0.003
        self.vf_lr = 1e-3

        self.ppo_clip = 0.5
        self.loss_entropy_coef = 0.0
        self.loss_vf_coef = 0.5

        self.normalize_adv = False

        self.init_nn()  
        dataset = [None]*8

        self.opt_policy = Adam(self.actor_net.parameters(), lr=self.policy_lr, eps=1e-5)

        self.opt_vf = RMSprop(self.vf_net.parameters(), lr=self.vf_lr)
        self.loss_vf_fn = torch.nn.MSELoss()


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

        self.actor_net = nn.Sequential(
            nn.Linear(self.n_state, self.actor_arch),
            nn.Tanh(),
            nn.Linear(self.actor_arch, self.actor_arch),
            nn.Tanh(),
            nn.Linear(self.actor_arch, self.actor_arch),
            nn.Tanh(),
            nn.Linear(self.actor_arch, self.n_disc_action),
        )


    def build_dataset_from_policy_dict(self):

        obs = self.env.reset()
        obs = obs[0]

        dataset_obs = []
        dataset_new_obs = []
        dataset_act = []
        dataset_rew = []
        dataset_end = []
        dataset_log_prob = []
        ep_rews = []

        for t in range(self.horizon):
            state = torch.from_numpy(obs)

            # Get Action
            action_dist = Categorical(logits=self.actor_net(state))
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action).sum(axis=-1)
            action_np = action.numpy()

            ## Apply Action into Env
            new_obs, reward, done, info, _ = self.env.step(action_np)

            dataset_obs.append(obs)
            dataset_new_obs.append(new_obs)
            dataset_act.append(action.detach())
            ep_rews.append(reward)
            dataset_end.append(done)
            dataset_log_prob.append(log_prob.detach())

            # If Done, reset environment
            if done:
                # print(done)
                obs = self.env.reset()
                obs = obs[0]
                dataset_rew.append(ep_rews)
                ep_rews = []
            else:
                obs = new_obs


        dataset_rew.append(ep_rews)

        dataset_obs = torch.tensor(np.array(dataset_obs), dtype=torch.float)
        dataset_new_obs = torch.tensor(np.array(dataset_new_obs), dtype=torch.float)
        dataset_act = torch.tensor(np.array(dataset_act), dtype=torch.float)
        dataset_rew_to_go = self.compute_rtgs(dataset_rew)
        dataset_end = torch.tensor(dataset_end, dtype=torch.float)
        dataset_log_prob = torch.tensor(dataset_log_prob, dtype=torch.float)


        return dataset_obs, dataset_new_obs, dataset_act, dataset_rew_to_go, dataset_end, dataset_log_prob


    def get_action(self, obs):
        action_probs = self.actor_net(torch.tensor(obs))
        return torch.argmax(action_probs)


    def compute_rtgs(self, batch_rews):
        """
			Compute the Reward-To-Go of each timestep in a batch given the rewards.

			Parameters:
				batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

			Return:
				batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
		"""
		# The rewards-to-go (rtg) per episode per batch to return.
		# The shape will be (num timesteps per episode)
        batch_rtgs = []

		# Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 # The discounted reward so far

			# Iterate through all rewards in the episode. We go backwards for smoother calculation of each
			# discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gae_gamma
                batch_rtgs.insert(0, discounted_reward)

		# Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs


    def learn(self):
        obs = self.env.reset()

        ## Training
        loss_train = []
        self.rew_train = []
        self.vf_train = []

        for ep in range(self.n_episodes):
            dataset_obs, dataset_new_obs, dataset_act, dataset_rew_to_go, dataset_end, dataset_log_prob = self.build_dataset_from_policy_dict()

            vf_value = self.vf_net(dataset_obs).squeeze()
            adv_batch = dataset_rew_to_go - vf_value.detach()

            adv_batch = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-10)

            loss_updates = []
            for _ in range(self.n_updates_per_ep):
                

                action_dist = Categorical(logits=self.actor_net(dataset_obs))
                log_prob_pi = action_dist.log_prob(dataset_act)

                ratio_pi = torch.exp(log_prob_pi - dataset_log_prob)

                lclip_p1 = ratio_pi * adv_batch
                lclip_p2 = adv_batch * torch.clamp(ratio_pi, 1 - self.ppo_clip, 1 + self.ppo_clip)
                lclip = -torch.min(lclip_p1,lclip_p2).mean()

                lentropy = -torch.mean(-log_prob_pi)

                loss_policy = lclip + self.loss_entropy_coef * lentropy
                loss_policy = lclip
            
                # Opt Policy
                self.opt_policy.zero_grad()
                loss_policy.backward()
                self.opt_policy.step()

            for _ in range(self.n_updates_per_ep):
                self.opt_vf.zero_grad()

                vf_value = self.vf_net(dataset_obs).squeeze()

                # # Opt VF
                loss_vf = self.loss_vf_fn(vf_value, dataset_rew_to_go)
                # self.opt_vf.zero_grad()
                loss_vf.backward()
                self.opt_vf.step()

                loss_updates.append(loss_vf.detach().numpy())

            loss_train.append(np.mean(np.array(loss_updates)))

            if ep % self.freq_eval == 0:
                # print(f'Loss {ep}')
                print(f'Loss ep {ep}: {loss_train[-1]:.3f} | Avg Reward: {dataset_rew_to_go.mean():.3f}')



cart_pole_ppo = CartPolePPO()
cart_pole_ppo.learn()
print(f'Learning finished')


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
