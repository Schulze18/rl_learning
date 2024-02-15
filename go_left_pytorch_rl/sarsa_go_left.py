from GoLeftEnv import GoLeftEnv

import torch


torch.manual_seed(1)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

grid_size = 10

n_states = 10
n_actions = 2
n_episodes = 1000

env = GoLeftEnv(grid_size = grid_size, max_steps = 50)

action_space = env.action_space

# Q = torch.randn(n_states, n_actions)
Q = torch.ones(n_states, n_actions) * 0
Q[0,:] = torch.zeros(1, n_actions)

alpha = 0.01
gamma = 1
epsilon = 0.01

eval_freq = 50

def epsilon_greedy(epsilon, Qs):
    p = torch.rand(1)

    if p < epsilon:
        index_action = torch.randint(0, 2, (1,))[0]
    else:
        index_action = torch.argmax(Qs)

    return index_action


for ep in range(n_episodes):
    # state = torch.randint(0, n_states)

    obs = env.reset(seed = 1)
    state = obs[0].astype(int)

    initial_action = epsilon_greedy(epsilon,Q[state,:])
    action = initial_action

    done = False
    while not done:
        obs, reward, done, info, _ = env.step(action)
        new_state = obs[0].astype(int)
        done = done | info
        
        # select action / e - greedy
        new_action = epsilon_greedy(epsilon,Q[state,:])

        Q[state, action] += alpha*(reward + gamma*Q[new_state, new_action] - Q[state, action])
        state = new_state
        action = new_action

    if ep % eval_freq == 0:
        print("Ep: ", ep)


print("Finished training")
print("Q: ")
print(Q)

## Eval Policy
n_eval = 15
for ev in range(n_eval):
    obs = env.reset(seed = 1)
    initia_state = obs[0].astype(int)
    state = initia_state
    done = False
    n_steps = 0

    while not done:
        
        # select action / e - greedy
        new_action = epsilon_greedy(epsilon,Q[state,:])

        obs, reward, done, info, _ = env.step(new_action)
        
        new_state = obs[0].astype(int)
        state = new_state
        done = done | info
        if info:
            print("Truncated")

        n_steps = n_steps + 1

    print(f'Eval: {ev}, Initial state: {initia_state}, n_steps: {n_steps}')
