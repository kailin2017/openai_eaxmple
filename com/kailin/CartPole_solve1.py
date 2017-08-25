import numpy
import gym

def evalute_given_parameter_by_sign(env,weight):

    observation = env.reset()
    total_reward = 0
    for t in range(1000):
        # 渲染函數為正常執行,若隱藏的話程式碼可以正常執行,但不會顯示畫面
        env.render()
        weighted_sum = numpy.dot(weight,observation)

        # 根據符號policy 選出action
        if weighted_sum > -1:
            action = 1
        else:
            action = 0

        observation,reward,done,info = env.step(action)
        total_reward += reward
        if done:
            print('gameover-',t)
            break
    return total_reward

def random_guess():
     env = gym.make('CartPole-v0')
     numpy.random.seed(10)

     best_reward = -100.0

     for i in range(1000):

         weight = best_reward + numpy.random.normal(0,0.01,4)

         cur_reward = evalute_given_parameter_by_sign(env,weight)
         if cur_reward > best_reward:
             best_reward = cur_reward
             best_weight = weight

         if best_reward==1000:
             break;
     print("best reward",best_reward)
     print("best weight",best_weight)

random_guess()