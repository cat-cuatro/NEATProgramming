import os
import sys
import gym
from gym import wrappers
import pickle
import neat
import numpy as np
import matplotlib.pyplot as plot

import visual


#MAX_FITNESS = 200
MAX_STEPS = 1000
MAX_GENERATIONS = 1000
GAME_TO_TEST = 'CartPole-v0'    # 0.02s step updates
#GAME_TO_TEST = 'MountainCar-v0'
filename = GAME_TO_TEST
path = "data/" + GAME_TO_TEST + "/"


def debugFunction():
    env = gym.make(GAME_TO_TEST)
    observation = env.reset()
    observation, reward, done, info = env.step(env.action_space.sample())
    print("***** Action and Observation Space for "+GAME_TO_TEST," *****")
    print("Observation Space: ", env.observation_space)
    print("Action space: ", env.action_space)
    print("Reward: :", reward)
    for steps in range(0,1000):
        env.render()
        observation, reward, done, info = env.step(2)

def selectAction(net_output, game):
    if(game == 'CartPole-v0'):
        if(net_output[0] > 0.5):
            action = 1
        else:
            action = 0
    elif(game == 'MountainCar-v0'):
        if(net_output[0] >= (2.0/3.0)):
            action = 2
        elif(net_output[0] >= (1.0/3.0) and net_output[0] < (2.0/3.0)):
            action = 1
        else:
            action = 0
    return action

def playAgent(winner, config, game):
    print(winner.fitness)
    # Save video of winner
    env_to_wrap = gym.make(game).env
    env = wrappers.Monitor(env_to_wrap, path, force=True)
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

def eval_genomes(genomes, config):
    nets = [] # the network for that environment
    envs = [] # environment for it
    ge = [] # genomes
    isDone = []
    observations = []
    i = 0
    a_reward = 0
    game = GAME_TO_TEST

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)                     # each env has its own net
        envs.append(gym.make(game).env) # create our game environment
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
                action = selectAction(nets[x].activate(observations[x]), game)
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

    pe = neat.ParallelEvaluator(4, eval_genomes)
    winner = p.run(eval_genomes)#, MAX_GENERATIONS)  ## uncomment when not debugging

#    debugFunction() ## for sampling the observation and action space of an OpenAI Game

    print('\nBest genome:\n{!s}'.format(winner))

    # Plot
    plot = None
    try:
        arg = sys.argv[2]
        plot = arg
    except:
        plot = False
    visual.plot(stats, GAME_TO_TEST, path, view=plot)
    visual.species(stats, GAME_TO_TEST + '-generations', path, view=plot)
    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visual.draw_net(config, winner, GAME_TO_TEST + "-nodes", path, view=plot, 
                    node_names=node_names,
                    show_disabled=True, prune_unused=False)
    visual.draw_net(config, winner, GAME_TO_TEST + "-nodes-disable", path, view=False, 
                    node_names=node_names,
                    show_disabled=False, prune_unused=False)
    visual.draw_net(config, winner, GAME_TO_TEST + "-nodes-pruned", path, view=False, 
                    node_names=node_names,
                    show_disabled=True, prune_unused=False)
    visual.draw_net(config, winner, GAME_TO_TEST + "-nodes-disable-pruned", path, view=False, 
                    node_names=node_names,
                    show_disabled=False, prune_unused=True)

    # Save winner
    with open(path + GAME_TO_TEST + '-Winner', 'wb') as f:
        pickle.dump(winner, f)

    playAgent(winner, config, GAME_TO_TEST)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config/config-nonspeciated")
    run(config_path)
