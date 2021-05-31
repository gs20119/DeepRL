
import numpy as np
import torch
from GridEnv.Gridworld import Gridworld
import torch.nn as nn
import random, copy
import matplotlib.pyplot as plt
from collections import deque

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
actions = { 0: 'u', 1: 'd', 2: 'l', 3: 'r' } # define action


def getState(game): # IMPORT STATE FROM GAME + ADD BIAS
    state = game.board.render_np().reshape(1, 64)
    noise = np.random.rand(1, 64) / 100.0
    return torch.from_numpy(state + noise).float().to(device)


def useBatch(batch): # DECOMPOSE BATCH INTO USEFUL ELEMENTS
    pS = torch.cat([ps for (ps, a, r, ns, d) in batch]).to(device)
    A = torch.Tensor([a for (ps, a, r, ns, d) in batch]).to(device)
    R = torch.Tensor([r for (ps, a, r, ns, d) in batch]).to(device)
    nS = torch.cat([ns for (ps, a, r, ns, d) in batch]).to(device)
    D = torch.Tensor([d for (ps, a, r, ns, d) in batch]).to(device)
    return pS, A, R, nS, D


def test_model(model, mode='static', display=True): # TEST TRAINED MODEL
    test = Gridworld(size=4, mode=mode)
    pState = getState(test)
    if display:
        print("Initial State:")
        print(test.display())

    for mv in range(15):
        Q = model(pState)
        A = np.argmax(Q.to('cpu').data.numpy())
        action = actions[A]

        test.makeMove(action)
        nState = getState(test)
        reward = test.reward()
        pState = nState

        if display:
            print('Move #%s - Taking action %s' % (mv, action))
            print(test.display())

        if reward != -1:
            if display: print("Game %s" % ('WIN' if reward > 0 else 'LOSE'))
            return reward > 0

    if display: print("Time OUT")
    return False
