import gymnasium as gym

from PPO import PPO

env = gym.make("CartPole-v1")

ppo_test = PPO(env, n_episodes = 200, batch_size = 500, 
               n_updates_per_ep = 20, ppo_clip=0.2,
               loss_vf_coef = 0.5, lambda_gae = 0.9)
ppo_test.learn()


env_eval = gym.make("CartPole-v1", render_mode="human")
obs = env_eval.reset()
obs = obs[0]
while True:
    action = ppo_test.get_action(obs)
    obs, reward, done, info, _ = env_eval.step(action.numpy())
    env_eval.render()
    # VecEnv resets automatically
    if done:
      obs = env_eval.reset()
      obs = obs[0]
