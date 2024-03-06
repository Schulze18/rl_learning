import gymnasium as gym

from PPO import PPO

import torch

flag_train = False
PATH = './half_cheetah_3000.pth'

if flag_train:
  # env = gym.make('HalfCheetah-v4', render_mode="human")
  env = gym.make('HalfCheetah-v4')

  ppo_test = PPO(env, n_episodes = 3000, batch_size = 1000, 
                n_updates_per_ep = 20, ppo_clip=0.2,
                loss_vf_coef = 0.5, lambda_gae = 0.9)
  ppo_test.learn()


  # SAVE
  torch.save({
              'policy_net_state_dict': ppo_test.policy_net.state_dict(),
              'vf_state_dict': ppo_test.vf_net.state_dict(),
              'opt_state_dict': ppo_test.opt_policy_vf.state_dict(),
              'log_std_state_dict': ppo_test.log_std,
              }, PATH)


##
load_data = torch.load(PATH)
env_eval = gym.make("HalfCheetah-v4", render_mode="human")
ppo_eval = PPO(env_eval)
ppo_eval.policy_net.load_state_dict(load_data['policy_net_state_dict'])
ppo_eval.vf_net.load_state_dict(load_data['vf_state_dict'])
ppo_eval.opt_policy_vf.load_state_dict(load_data['opt_state_dict'])
ppo_eval.log_std = load_data['log_std_state_dict']

# env_eval = gym.make("HalfCheetah-v4", render_mode="human")
obs = env_eval.reset()
obs = obs[0]
while True:
    action = ppo_eval.get_action(obs)
    obs, reward, done, truncated, _ = env_eval.step(action.numpy())
    env_eval.render()
    if done or truncated:
      obs = env_eval.reset()
      obs = obs[0]