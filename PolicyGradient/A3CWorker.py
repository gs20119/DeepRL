
from ActorCritic import *
from ActorCriticRun import *
import torch.multiprocessing as mp
import gym
from torch import optim

Network = ActorCritic()
Network.share_memory()
counter = mp.Value('i',0) # 'i',0 == integer from zero
params = { # params dictionary
    'epochs':1000,
    'n_workers':7,
}

processes = []
for i in range(params['n_workers']):
    p = mp.Process(target=train, args=(Network, params))
    p.start() # start train(Network, params) with multiprocessing
    processes.append(p)

for p in processes: p.join()
for p in processes: p.terminate()
