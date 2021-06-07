
from ActorCritic import *
import torch.multiprocessing as mp
import gym
from torch import optim
import torch

MasterNode = ActorCritic()
MasterNode.share_memory()
processes = []

params = { 'epochs':1000, 'n_workers':7, } # params dictionary
counter = mp.Value('i',0) # i means

def run_episode(worker_env, worker_model):
    state = torch.from_numpy(worker_env.env.state).float()
    values, logprobs, rewards = [], [], []
    done = False
    j = 0
    while not done:
        j += 1
        policy, value = worker_model(state)
        values.append(value)
        logits = policy.view(-1)
        action_dist = torch.distributions.Categorical(logits=logits)
        action = action_dist.sample()
        logprob_ = policy.view(-1)[action]
        logprobs.append(logprob_)
        state_, _, done, info = worker_env.step(action.detach().numpy())
        state = torch.from_numpy(state_).float()
        reward = -10 if done else 1.0
        if done: worker_env.reset()
        rewards.append(reward)
    return values, logprobs, rewards

def update_params(worker_opt, values, logprobs, rewards, clc=0.1, gam=0.95):
    rewards = torch.Tesor(rewards).flip(dims=(0,)).view(-1)
    logprobs = torch.stack(logprobs).flip(dims=(0,)).view(-1)
    values = torch.stack(values).flip(dims=(0,)).view(-1)
    actor_loss = -1*logprobs * (Returns - values.detach())
    critic_loss = torch.pow(values - Returns, 2)
    return actor_loss, critic_loss, len(rewards)

def worker(t, worker_model, counter, params):
    worker_env = gym.make('CartPole-v1')
    worker_env.reset()
    worker_opt = optim.Adam(lr=1e-4, parames=worker_model.parameters())
    worker_opt.zero_grad() # gradient reset
    for i in range(params['epochs']):
        worker_opt.zero_grad()
        values, logprobs, rewards = run_episode(worker_env, worker_model)
        loss_actor, loss_critic, eplen = update_params(worker_opt, values, logprobs, rewards)
        counter.value += 1

for i in range(params['n_workers']):
    p = mp.Process(target=worker, arges=(i,MasterNode,counter,params))
    p.start()
    processes.append(p)

for p in processes: p.join()
for p in processes: p.terminate()