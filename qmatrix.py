import random
import numpy as np

class qarray(object):
    def __init__(self, actionSpace, numberOfActions, boardGame1, gameType):
        random.seed()
        self.game = boardGame1
        self.gametype = gameType
        self.currentState = None
        self.previousState = None
        self.qtable = []  # current state-action weight pairs, ie. [[0, 0, 'X', 0, 0 'O', . .], [.01, .032, 0, -.1 . .]]
        self.storedStates = 0
        self.initialWeights = []
        self.numberOfActions = numberOfActions
        self.actions = actionSpace # space of actions should be passed in as a list
        for _ in range(0, numberOfActions):
            self.initialWeights.append(0) # initialize weights to 0
    
    def addState(self, currentState):
        index = self.storedStates
        self.qtable.append([currentState, self.initialWeights])
        self.storedStates += 1
        return index

    def stateExists(self, currentState):
        stateExists = False
        for x in range(0, self.storedStates):
            if(self.qtable[x][0] == list(currentState)):
                index = x
                stateExists = True
        if(stateExists == False):
            index = self.addState(currentState)
        return index

    def chooseAction(self, epsilon, epoch, currentState, currentIndex, player):
        dice = random.randint(0,100)
        validAction = False
        firstAttempt = True
        while(validAction == False):
            if(firstAttempt == True): # pick a random 'start point' to avoid local max/mins
                start = random.randint(0, (self.numberOfActions-1))
                highestValue = self.qtable[currentIndex][1][start]
                action = start # this is the index for the location where this action is stored
                firstAttempt = False
            else:
                highestValue = self.qtable[currentIndex][1][0]
                action = 0
            if(dice <= (100*epsilon)):
                action = random.randint(0, (self.numberOfActions-1))
            else: # otherwise, make a greedy decision
                for x in range(0,self.numberOfActions):
                    if(highestValue < self.qtable[currentIndex][1][x]):
                        highestValue = self.qtable[currentIndex][1][x]
                        action = x
            if(self.gametype == 'tictactoe'):
                validAction = self.game.checkMoveValidity(self.actions[action], player)

            if(validAction == False and self.gametype == 'tictactoe'):
                self.qtable[currentIndex][1][action] -= 0.1 #invalid moves should be negative

            if(self.gametype != 'tictactoe'):
                validAction = True

        return self.actions[action], action # return the action taken, and the index used to find it

    def isGameDone(self):
        return self.game.checkForVictory()

    def retrieveMaxQVal(self, currentState):
        index = self.stateExists(currentState)
        highestValue = self.qtable[index][1][0]
        for x in range(0, self.numberOfActions):
            if(highestValue <= self.qtable[index][1][x]):
                highestValue = self.qtable[index][1][x]
        return highestValue

    def updateWeight(self, pastState, pastIndex, currentState, actionIndex, learning_rate, diff):
        maxQ = self.retrieveMaxQVal(currentState)
        pastWeight = self.qtable[pastIndex][1][actionIndex]
        self.qtable[pastIndex][1][actionIndex] = pastWeight + learning_rate*(diff + (0.9*maxQ) - pastWeight)

    def printStatesEncountered(self):
        print(self.storedStates, " unique states encountered.")

## Functions built around the tic-tac-toe game ##

    def examineBoard(self):
        return self.game.buildStateString()

    def randomAgent(self, player):
        validAction = False
        while(validAction == False):
            action = random.randint(0,8)
            validAction = self.game.checkMoveValidity(self.actions[action], player)
        self.performAction(self.actions[action], player)

    def rewardAgent(self, winner):
        return self.game.resetBoard(winner)

    def displayBoard(self):
        self.game.printSimpleGameState()

    def userIsPlaying(self, action):
        validAction = self.game.checkMoveValidity(self.actions[action], 'O')
        while(validAction == False):
            action = int(input("Illegal or bad input. Pick a tile"))
            validAction = self.game.checkMoveValidity(self.actions[action], 'O')
        self.performAction(self.actions[action], 'O')

    def performAction(self, action, player):
        # the action to be performed needs to be verified as valid beforehand
        if(player == 'X'):
            self.game.crossTurn(action)
        elif(player == 'O'):
            self.game.circleTurn(action)
