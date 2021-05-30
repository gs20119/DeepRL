
from utils import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

l1, l2, l3, l4 = 64, 150, 100, 4

model = nn.Sequential( # network architecture
    nn.Linear(l1,l2), nn.ReLU(),
    nn.Linear(l2,l3), nn.ReLU(),
    nn.Linear(l3,l4)
)

model = model.to(device) # gpu setting
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
gam, eps = 0.9, 0.3

actions = { 0: 'u', 1: 'd', 2: 'l', 3: 'r' } # define action

episodes = 10000
mem_size = 1000
batch_size = 200
replay = deque(maxlen=mem_size) # +USE REPLAY MEMORY
max_mv = 50
losses = []


for i in range(episodes): # DQN Learning with EXPERIENCE REPLAY
    if i % 100 == 0: print(i)
    game = Gridworld(size=4, mode='random') # create new game (env)
    pState = getState(game) # import state of game

    for mv in range(max_mv): # 1 iteration = 1 game
        Q = model(pState) # objective : s -> neural network = table of Q(s,a)
        if random.random() < eps: A = np.random.randint(0,4)
        else: A = np.argmax(Q.to('cpu').data.numpy())
        action = actions[A] # choose action : eps-greedy

        game.makeMove(action) # move -> get reward & state
        nState = getState(game)
        reward = game.reward()

        exp = (pState, A, reward, nState, (reward > 0))
        replay.append(exp) # CREATE EXPERIENCE THEN ADD TO REPLAY

        if len(replay) > batch_size: # SELECT BATCH IN REPLAY TO TRAIN
            batch = random.sample(replay, batch_size)
            pState_, A_, reward_, nState_, win_ = useBatch(batch)

            prevQ = model(pState_)
            with torch.no_grad(): newQ = model(nState_)

            Y = reward_ + gam * ((1 - win_) * torch.max(newQ, dim=1)[0]) # using Q-learning formula
            Y = Y.detach().to(device)
            X = prevQ.gather(index=A_.long().unsqueeze(dim=1), dim=1).squeeze() # samples X & Y

            loss = loss_fn(X,Y)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())
            optimizer.step() # optimizing process

        pState = nState
        if reward != -1: break

plt.plot(np.arange(0,len(losses)), losses)
plt.show()


wins = 0
for i in range(1000):
    win = test_model(model, mode='random', display=False)
    if win: wins += 1

win_rate = float(wins) / float(1000)
print("Win ratio - %s" % win_rate)
