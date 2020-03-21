import os
import gym
from gym import wrappers
import pickle
import neat
import numpy as np
import matplotlib.pyplot as plot
from actions import ActionSelector as act
from DebugTools import debug as dt
from testing import play as testAgent

import visual

path = "data/"

MAX_STEPS = 300
MAX_GENERATIONS = 50
WINNER_LIST = []
GAME_TO_TEST = 'CartPole-v0'
#GAME_TO_TEST = 'MountainCar-v0'

def eval_genomes(genomes, config):
    nets = [] # the network for that environment
    envs = [] # environment for it
    ge = [] # genomes
    isDone = []
    observations = []
    i = 0
    a_reward = 0
    game = GAME_TO_TEST
    bestFitness = 0

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)                     # each env has its own net
        envs.append(gym.make('CartPole-v0').env) # create our game environment
        g.fitness = 0
        ge.append(g)                         # and its own genome
        observations.append(envs[i].reset()) # keep track of this observation
        isDone.append(False)
        i += 1

    # Timestep == 0.02s
    for steps in range(0, MAX_STEPS): # up to MAX_STEPS steps
        # take 1 step in every cartpole environment (there are len(envs) of them)
        for x in range(0, len(envs)):
           if(isDone[x] == False):
                action = act.selectAction(nets[x].activate(observations[x]), game)
                # perform an action based on our observation, and update our observation
                observations[x], a_reward, isDone[x], info = envs[x].step(action)
                ge[x].fitness += a_reward  # update the reward
                if(bestFitness < ge[x].fitness):
                    bestFitness = ge[x].fitness
                    i = x
                # if we're done, we need to stop using that agent,
                # isDone tells us if that agent has failed the CartPole
        if(len(envs) == 0): # if all agents are done, I should stop early.
            break
    WINNER_LIST.append(ge[x])


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    pe = neat.ParallelEvaluator(10, eval_genomes)
    winner = p.run(eval_genomes, MAX_GENERATIONS)  ## uncomment when not debugging
    WINNER_LIST.append(winner)

#    dt.debugFunction() ## for sampling the observation and action space of an OpenAI Game

    print('\nBest genome:\n{!s}'.format(winner))

    # Save winner
    with open(path + GAME_TO_TEST + '-Winner', 'wb') as f:
        pickle.dump(winner, f)

#    testAgent.playAgent(winner, config, GAME_TO_TEST, MAX_STEPS) # Play the best fitness agent
    print(len(WINNER_LIST))
    testAgent.playAllAgents(WINNER_LIST, config, GAME_TO_TEST, MAX_STEPS) # Play the best fitness agent from each iteration


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
