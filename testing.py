import gym
from gym import wrappers
import neat
import os
from actions import ActionSelector as act

class play(object):
    def playAgent(winner, config, game, maxSteps):
        print(winner.fitness)
        env_to_wrap = gym.make(game).env
        env = wrappers.Monitor(env_to_wrap, "data/", force=True, video_callable=lambda episode_id: True)
        observation = env.reset()
        net = neat.nn.FeedForwardNetwork.create(winner, config)
        a_reward = 0
        sum = 0
        done = False
        for steps in range(0, maxSteps):
            env.render()                                           # Render must be in the loop
            action = act.selectAction(net.activate(observation), game) # convert net output into an action acceptable by OpenAI's action API
            observation, a_reward, done, info = env.step(action)   # standard API step call, returns 4 datums
            sum += a_reward
            if done: # terminate if agent is failed/done
                print("Broke at step: ", steps)
                break
            env.close()

    def playAllAgents(winnerList, config, game, maxSteps):
        env_to_wrap = gym.make(game).env
        env = wrappers.Monitor(env_to_wrap, "data/MountainCar/", force=True, video_callable=lambda episode_id: True)
        sum= 0
        for agent in winnerList:
            observation = env.reset()
            net = neat.nn.FeedForwardNetwork.create(agent, config)
            for steps in range(0, maxSteps):
                env.render()
                action = act.selectAction(net.activate(observation), game)
                observation, a_reward, done, info = env.step(action)
                sum += a_reward
                if done:
                    print("Broke at step: ", steps)
                    break
            env.close()




