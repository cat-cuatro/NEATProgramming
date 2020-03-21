class ActionSelector(object):
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




