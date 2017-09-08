import numpy
import pandas

class QLearningTable:
    def __init__(self,actions,rate=0.1,gamma=0.9,epsilon=0.9):
        self.actions = actions
        self.rate = rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.qtable = pandas.DataFrame(columns=self.actions)

    def checkQTable(self,row):
        if row not in self.qtable.index:
            self.qtable = self.qtable.append(
                pandas.Series(
                    [0] * len(self.actions),
                    index=self.qtable.columns,
                    name=row,
                )
            )

    def learning(self,state,action,reward):
        self.checkQTable(state)
        q_predict = self.qtable.ix[state, action]
        q_target = reward + self.gamma * self.qtable.ix[state, :].max()  # next state is not terminal
        self.qtable.ix[state, action] += self.rate * (q_target - q_predict)

    def chooseAction(self,state):
        self.checkQTable(state)
        if numpy.random.uniform() < self.epsilon:
            # choose best action
            action_temp = self.qtable.ix[state, :]
            action_temp = action_temp.reindex(numpy.random.permutation(action_temp.index))
            action = action_temp.argmax()
        else:
            # choose random action
            action = numpy.random.choice(self.actions)
        return action