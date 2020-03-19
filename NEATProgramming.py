import matplotlib.pyplot as plot
import numpy as np
import gym
import neat
import os

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
        envs.append(gym.make('CartPole-v0')) # create our game environment
        g.fitness = 0
        ge.append(g)                         # and its own genome
        observations.append(envs[i].reset()) # keep track of this observation
        isDone.append(False)
        i += 1


    for steps in range(0, 200): # up to 200 steps
        for x in range(0, len(envs)): # take 1 step in every cartpole environment (there are len(envs) of them)
           if(isDone[x] == False):
                action = nets[x].activate(observations[x])
                if(action[0] > 0.5):
                    action = 1
                else:
                    action = 0
                observations[x], a_reward, isDone[x], info = envs[x].step(action) # perform an action based on our observation, and update our observation
                ge[x].fitness += a_reward  # update the reward
                # if we're done, we need to stop using that agent, isDone tells us if that agent has failed the CartPole
        if(len(envs) == 0): # if all agents are done, I should stop early.
            break

def run(config_path):
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
        p = neat.Population(config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)

        winner = p.run(main,50)
        print('\nBest genome:\n{!s}'.format(winner))

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)