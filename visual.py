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
    if view == True:
        plt.show()

    plt.savefig(path + filename)
    plt.close()

def species():
    pass


