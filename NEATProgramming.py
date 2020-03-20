import os
import gym
import pickle
import neat
import numpy as np
import matplotlib.pyplot as plot

import visual

path = "data/"
MAX_FITNESS = 1000
MAX_STEPS = 1000
MAX_GENERATIONS = 150

def selectAction(net_output, game):
    if(game == 'CartPole-v0'):
        if(net_output[0] > 0.5):
            action = 1
        else:
            action = 0

    return action

def playAgent(winner, config, game):
    print(winner.fitness)
    env = gym.make(game).env
    observation = env.reset()
    net = neat.nn.FeedForwardNetwork.create(winner, config)
    a_reward = 0
    sum = 0
    done = False
    for steps in range(0, MAX_STEPS):
        env.render()                                           # Render must be in the loop
        action = selectAction(net.activate(observation), game) # convert net output into an action acceptable by OpenAI's action API
        observation, a_reward, done, info = env.step(action)   # standard API step call, returns 4 datums
        sum += a_reward
        if done: # terminate if agent is failed/done
            print("Broke at step: ", steps)
            break
    env.close()


def main(genomes, config):
    nets = [] # the network for that environment
    envs = [] # environment for it
    ge = [] # genomes
    isDone = []
    observations = []
    i = 0
    a_reward = 0

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)                     # each env has its own net
        envs.append(gym.make('CartPole-v0').env) # create our game environment
        g.fitness = 0
        ge.append(g)                         # and its own genome
        observations.append(envs[i].reset()) # keep track of this observation
        isDone.append(False)
        i += 1


    for steps in range(0, MAX_STEPS): # up to MAX_STEPS steps
        # take 1 step in every cartpole environment (there are len(envs) of them)
        for x in range(0, len(envs)):
           if(isDone[x] == False):
                action = nets[x].activate(observations[x])
                if(action[0] > 0.5):
                    action = 1
                else:
                    action = 0
                # perform an action based on our observation, and update our observation
                observations[x], a_reward, isDone[x], info = envs[x].step(action)
                ge[x].fitness += a_reward  # update the reward
                # if we're done, we need to stop using that agent,
                # isDone tells us if that agent has failed the CartPole
        if(len(envs) == 0): # if all agents are done, I should stop early.
            break

def run(config_path):
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        winner = p.run(main, MAX_GENERATIONS)
        print('\nBest genome:\n{!s}'.format(winner))
#        print(MAX_STEPS, MAX_FITNESS)

        # Plot
        visual.plot(stats, 'CartPole-v0', view=True)

        # Save winner
        with open(path + 'CartPoleWinner', 'wb') as f:
            pickle.dump(winner, f)

        playAgent(winner, config, 'CartPole-v0')

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
