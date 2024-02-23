from PPO import PPO
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
ppo_teste = PPO(env, n_episodes = 100, batch_size = 500)
ppo_teste.learn()
