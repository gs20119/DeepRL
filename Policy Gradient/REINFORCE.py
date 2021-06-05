
import gym
import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

l1, l2, l3 = 4, 150, 2

model = nn.Sequential( # network architecture
    nn.Linear(l1,l2), nn.LeakyReLU(),
    nn.Linear(l2,l3), nn.Softmax(),
)

model = model.to(device)
env = gym.make('CartPole-v0')
optimizer = torch.optim.Adam(model.parameters(), lr=0.0009)
max_time, episodes = 200, 500
gam = 0.99
score = []

def discount(rewards, gamma): # full discount to reward batch
    length = len(rewards)
    result = torch.pow(gamma,torch.arange(length).float()) * rewards
    result /= result.max() # Normalize into range [0,1]
    return result

for i in range(episodes):
    pState = env.reset()
    done = False
    transitions = []

    for t in range(max_time):
        policy = model(torch.from_numpy(pState).float().to(device))
        action = np.random.choice(np.array([0,1]), p=policy.data.numpy())
        nState, _, done, info = env.step(action)
        transitions.append((pState, action, t+1))
