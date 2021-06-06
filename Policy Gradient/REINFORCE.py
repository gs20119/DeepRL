
import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

l1, l2, l3 = 4, 150, 2

model = nn.Sequential( # network architecture
    nn.Linear(l1,l2), nn.LeakyReLU(),
    nn.Linear(l2,l3), nn.Softmax(),
)

model = model.to(device) # gpu setting
env = gym.make('CartPole-v0') # import environment
optimizer = torch.optim.Adam(model.parameters(), lr=0.0009)
max_time, max_episode = 200, 500
scores = []


def discount(rewards, gamma=0.99): # full discount to reward batch
    length = len(rewards)
    result = torch.pow(gamma,torch.arange(length).float().to(device)) * rewards
    result /= result.max() # Normalize into range [0,1]
    return result

def useBatch(episode): # decompose episode into useful elements
    S = torch.Tensor([s for (s, a, r) in episode]).to(device)
    A = torch.Tensor([a for (s, a, r) in episode]).to(device)
    R = torch.Tensor([r for (s, a, r) in episode]).flip(dims=(0,)).to(device)
    R = discount(R)
    return S, A, R


for i in range(max_episode): # REINFORCE Learning - Policy Gradient
    pState = env.reset() # create new game (env)
    done = False
    episode = [] # REINFORCE is a type of episodic learning. Save transitions

    for t in range(max_time): # play game.
        env.render()
        policy = model(torch.from_numpy(pState).float().to(device)) # s -> Network = policy
        action = np.random.choice(np.array([0,1]), p=policy.to('cpu').data.numpy())
        nState, _, done, info = env.step(action) # choose action randomly according to policy
        episode.append((pState, action, t+1)) # add events to episode
        pState = nState
        if done: break

    score = len(episode) # score = how long did it survived
    scores.append(score)

    state_, action_, reward_ = useBatch(episode) # extract state, action, reward, policy batch
    policy_ = model(state_)
    policy_ = policy_.gather(index=action_.long().view(-1,1), dim=1).squeeze()

    loss = -1 * torch.sum(reward_ * torch.log(policy_)) # we have a formula for this (gradient)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # optimizing process

mov20 = [np.mean(scores[i:i+20]) for i in np.arange(len(scores)-20)] # moving average
plt.plot(np.arange(len(mov20)), mov20)
plt.show()
