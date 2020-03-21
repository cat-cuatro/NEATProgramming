import gym
from gym import wrappers
import neat
import os
from actions import ActionSelector as act
import qmatrix as qm
import DataLogger as de
import numpy as np


def roundState(currentState):
    for x in range(len(currentState)):
        currentState[x] = round(currentState[x],1)
    return currentState

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

    def trainQMatrix(qLearn1, epochs, epsilon, learning_rate, dataLog, testing, game):
        env = gym.make(game).env
        epsilonCounter = 0
        sum = 0
        prevSum = 0
        if(testing == False):
            steps = 500
        else:
            steps = 200

        for i in range(0, epochs):
            currentState = roundState(list(env.reset()))

            for t in range(0,steps):
                if(testing == True):
                    env.render()
                done = False
                reward = 0
                currentIndex = qLearn1.stateExists(currentState) # check if state exists, if it doesn't, then add it
                action, actionIndex = qLearn1.chooseAction(epsilon, i, currentState, currentIndex, None)
                observation, reward, done, info = env.step(action)
                observation = roundState(list(observation))

                pastState = currentState
                pastIndex = currentIndex
                currentState = observation
                sum += reward
    #            currentEpochReward += reward

                if(testing == False):
                    qLearn1.updateWeight(pastState, pastIndex, currentState, actionIndex, learning_rate, reward)
                if done:
                    if(testing == False):
                        break
                    else:
                        done = False

            if((i+1) % 100 == 0):
                currentEpochReward = (sum-prevSum)/100.0
                prevSum = sum
                print(i+1, " epochs processed ..")
                print("Reward Total: ", sum)
                print("Avg this epoch: ", currentEpochReward)

            epsilonCounter += 1
            if(epsilonCounter == 100 and epsilon > 0.05 and testing == False):
                epsilon = epsilon * 0.95
                epsilonCounter = 0
                print(epsilon) # Display current chance for stochastic behavior
    #                testAgent(qLearn1, 100, (i+1), dataLog, False)
            elif(epsilonCounter == 100 and epsilon <= 0.05 and testing == False):
                epsilonCounter = 0
                print(epsilon)
    #                testAgent(qLearn1, 100, (i+1), dataLog, False)

        print("Total reward received by agent: ", sum)



