
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
gam, eps = 0.9, 1.0

actions = { 0: 'u', 1: 'd', 2: 'l', 3: 'r' } # define action

episodes = 1000
max_mv = 50
losses = []


for i in range(episodes): # DQN Learning
    if i % 100 == 0: print(i)
    game = Gridworld(size=4, mode='static') # create new game (env)
    pState = getState(game) # import state of game
    eps = max(0.1, eps - (1/episodes))

    for mv in range(max_mv): # 1 iteration = 1 game
        Q = model(pState) # objective : s -> neural network = table of Q(s,a)
        if random.random() < eps: A = np.random.randint(0,4)
        else: A = np.argmax(Q.to('cpu').data.numpy())
        action = actions[A] # choose action : eps-greedy

        game.makeMove(action) # move -> get reward & state
        nState = getState(game)
        reward = game.reward()

        with torch.no_grad(): newQ = model(nState)
        maxQ = torch.max(newQ) # using Q-learning formula

        Y = reward + (gam * maxQ) if reward == -1 else reward
        Y = torch.Tensor([Y]).detach()
        X = Q.squeeze()[A] # samples X & Y

        loss = loss_fn(X.to(device), Y.to(device))
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step() # optimizing process

        pState = nState
        if reward != -1: break

plt.plot(np.arange(0,len(losses)), losses)
plt.show()

