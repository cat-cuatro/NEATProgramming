# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#
# Sources:
#   Some code borrowed from the visualize module by @CodeReclaimers, author
#   of neat-python:
#   https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/visualize.py
#
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


import graphviz
import matplotlib.pyplot as plt
import numpy as np

path = "data/"

def plot(stats, filename, view=False):

    gen = range(len(stats.most_fit_genomes))
    best_fit = [c.fitness for c in stats.most_fit_genomes]
    avg_fit = np.array(stats.get_fitness_mean())
    std_fit = np.array(stats.get_fitness_stdev())

    plt.plot(gen, best_fit, label='best')
    plt.plot(gen, avg_fit, label='average')
    plt.plot(gen, std_fit, label='std')

    plt.title("Population Averages and Best fit")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="upper left")

    plt.savefig(path + filename)
    if view == True:
        plt.show()
    plt.close()

def species(stats, filename, view=False):

    species_sizes = stats.get_species_sizes()    
    num_gen = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_gen), *curves)

    plt.ylabel("Size per species")
    plt.xlabel("Generations")

    plt.savefig(path + filename)
    if view:
        plt.show()
    plt.close()

