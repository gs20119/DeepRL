
import torch
from torch import nn
from torch import optim
import numpy as np
from torch.nn import functional as F
import gym

class ActorCritic(nn.Module): # How to make own network without Sequential
    def __init__(self): # init - define layers(structure)
        super(ActorCritic, self).__init__()
        self.l1 = nn.Linear(4,25)
        self.l2 = nn.Linear(25,50)
        self.l3_actor = nn.Linear(50,2)
        self.l3_critic = nn.Linear(50,25)
        self.l4_critic = nn.Linear(25,1)

    def forward(self,x): # forward - make 'y' for the given input 'x'
        x = F.normalize(x,dim=0) # normalize - makes the input more stable
        y = F.relu(self.l1(x))
        y = F.relu(self.l2(y))
        actor = F.log_softmax(self.l3_actor(y),dim=0) # actor output : policy
        critic = F.relu(self.l3_critic(y.detach())) # detach = disable backprop to y
        critic = F.tanh(self.l4_critic(critic)) # critic output : -1~1
        return actor, critic

