import gym
class debug(object):
    def debugFunction(game):
        env = gym.make(game)
        observation = env.reset()
        observation, reward, done, info = env.step(env.action_space.sample())
        print("***** Action and Observation Space for "+GAME_TO_TEST," *****")
        print("Observation Space: ", env.observation_space)
        print("Action space: ", env.action_space)
        print("Reward: :", reward)
        for steps in range(0,1000):
            env.render()
            observation, reward, done, info = env.step(2)
    def menu():
        print("***** MENU *****")
        print("1. Run NEAT")
        print("2. Run QTable")

    def envChoice():
        print("***** Environment *****")
        print("1. Cartpole")
        print("2. Mountain Car")