import gym
env = gym.make('CartPole-v0')

for i_episode in range(20):
    state = env.reset()
    for t in range(500):
        env.render()
        print(state)
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} time steps".format(t+1))
            break

env.close()
