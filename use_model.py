from unityagents import UnityEnvironment
import numpy as np
from collections import deque
from dqn_agent import Agent
import matplotlib.pyplot as plt
import torch


env = UnityEnvironment(file_name="../Banana_Linux/Banana.x86_64")
num_episodes = 10
#Set epsilon value at 0 since we're not exploring
eps = 0

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

agent = Agent(state_size, action_size, 0)
#Load the model
if torch.cuda.is_available():
    map_location=lambda storage, loc: storage.cuda()
else:
    map_location='cpu'

agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth',map_location=map_location))

for i in range(num_episodes):
    # for i_episode in range(1, n_episodes + 1):
    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    score = 0
    while True:
        action = agent.act(state, eps)
        env_info = env.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations[0]  # get the next state
        reward = env_info.rewards[0]  # get the reward
        done = env_info.local_done[0]  # see if episode has finished
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break

    print('\rEpisode {}\tScore: {:.2f}'.format(i, score))

env.close()