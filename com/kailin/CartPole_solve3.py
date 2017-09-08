from com.kailin.q_learning import QLearningTable
import gym

env = gym.make('CartPole-v0')
for x in range(1000):
    env.reset()
    Q = QLearningTable(actions=list(range(2)))
    for y in range(1000):
        env.render()
        action = Q.chooseAction(y)
        observation, reward, done, info = env.step(action)
        Q.learning(y,action,reward)
        if done:
            print('Game',x,'counter',y)
            break