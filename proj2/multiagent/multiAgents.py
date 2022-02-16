# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from argparse import Action
from platform import node
import this

from numpy import number, who
from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0
        currFood = currentGameState.getFood()
        currPos = currentGameState.getPacmanPosition()
        score += 5 * (currentGameState.getNumFood() - successorGameState.getNumFood())
        score += 30 * (len(currentGameState.getCapsules()) - len(successorGameState.getCapsules()))
        for i in range(len(newGhostStates)):
            score += self.ghostChase(newGhostStates[i], newScaredTimes[i], newPos, currPos)
        score += self.getClosestFood(newPos, newFood)
        return score
    
    def ghostChase(self, ghost, scaredTime, newPos, oldPos):
        score = 0
        oldDis = self.distanceFromGhost(ghost, oldPos)
        newDis = self.distanceFromGhost(ghost, newPos)
        if scaredTime > 4:
            score = oldDis - newDis
        else:
            #naive approach
            score = -(12 / (0.1 + newDis))
        return score
    
    def distanceFromGhost(self, ghost, pos):
        ghostPos = ghost.getPosition()
        return self.manhatttanDistance(pos, ghostPos)
    def getClosestFood(self, nP, food):
        closest = 9999
        foods = food.asList()
        for f in foods:
            closest = min(closest, self.manhatttanDistance(nP, f))
        return 4 / closest
    def manhatttanDistance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs (p1[1] - p2[1])


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimax_decision(gameState, self.depth)

    def minimax_decision(self, gameState, depth):
        bestScore = -99999
        bestAction = None
        for action in gameState.getLegalActions(0):
            score = self.minimax_value(gameState.generateSuccessor(0, action), "MIN", depth)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction
    def minimax_value(self, gameState, player, depth):
        if depth == 0 or isTerminal(gameState):
            return self.evaluationFunction(gameState)
        if player == "MAX":
            bestScore = -99999
            for action in gameState.getLegalActions(0):
                score = self.minimax_value(gameState.generateSuccessor(0, action), "MIN", depth)
                if score > bestScore:
                   bestScore = score
            return bestScore
        if player == "MIN":
            bestScore = 99999
            for outcome in allGhostReactions(gameState):
                score = self.minimax_value(outcome, "MAX", depth - 1)
                if score < bestScore:
                   bestScore = score
            return bestScore

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alphaBeta_decision(gameState, self.depth)

    def alphaBeta_decision(self, gameState, depth):
        bestScore = -99999
        alpha = -999999
        beta = 999999
        bestAction = None
        for action in gameState.getLegalActions(0):
            score = self.alphaBeta_value(gameState.generateSuccessor(0, action), "MIN", depth, alpha, beta)
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, bestScore)
        return bestAction

    def alphaBeta_value(self, gameState, player, depth, alpha, beta):
        if depth == 0 or isTerminal(gameState):
            return self.evaluationFunction(gameState)
        if player == "MAX":
            bestScore = -99999
            for action in gameState.getLegalActions(0):
                bestScore = max(bestScore, self.alphaBeta_value(gameState.generateSuccessor(0, action), "MIN", depth, alpha, beta))
                if bestScore > beta:
                    return bestScore
                alpha = max(alpha, bestScore)
            return bestScore
        if player == "MIN":
            bestScore = 99999
            movesToSuccessor = allPossibleMoveCombinations(getAllGhostMoves(gameState))
            for moves in movesToSuccessor:
                successorState = getNextStateForGhostMoves(gameState, moves)
                bestScore = min(bestScore, self.alphaBeta_value(successorState, "MAX", depth - 1, alpha, beta))
                if bestScore < alpha:
                   return bestScore
                beta = min(beta, bestScore)
            return bestScore

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax_decision(gameState, self.depth)

    def expectimax_decision(self, gameState, depth):
        bestScore = -99999
        bestAction = None
        for action in gameState.getLegalActions(0):
            score = self.expectimax_value(gameState.generateSuccessor(0, action), "MIN", depth)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction
    def expectimax_value(self, gameState, player, depth):
        if depth == 0 or isTerminal(gameState):
            return self.evaluationFunction(gameState)
        if player == "MAX":
            bestScore = -99999
            for action in gameState.getLegalActions(0):
                score = self.expectimax_value(gameState.generateSuccessor(0, action), "MIN", depth)
                if score > bestScore:
                   bestScore = score
            return bestScore
        if player == "MIN":
            totalScore = 0
            nodeNum = 0
            for outcome in allGhostReactions(gameState):
                score = self.expectimax_value(outcome, "MAX", depth - 1)
                totalScore += score
                nodeNum += 1
            return totalScore / nodeNum

def allGhostReactions(gameState):
    allReactions = []
    for moves in allPossibleMoveCombinations(getAllGhostMoves(gameState)):
        allReactions.append(getNextStateForGhostMoves(gameState, moves))
    return allReactions

def getAllGhostMoves(gameState):
    moves = []
    numberOfGhosts = gameState.getNumAgents() - 1
    for i in range(numberOfGhosts):
        moves.append(gameState.getLegalActions(i + 1))
    return moves

def allPossibleMoveCombinations(allMoves):
    if not allMoves:
        return []
    answer = []
    firstGhostMoves = allMoves[0]
    for move in firstGhostMoves:
        instance = [move]
        for rest in allPossibleMoveCombinations(allMoves[1:]):
            whole = instance + rest
            answer.append(whole)
        if not allPossibleMoveCombinations(allMoves[1:]):
            answer.append(instance)
    return answer

def isTerminal(gameState):
    return gameState.isWin() or gameState.isLose()

def getNextStateForGhostMoves(gameState, moves):
    ghostIndex = 1
    s = gameState
    for move in moves:
        if not isTerminal(s):
            s = s.generateSuccessor(ghostIndex, move)
        ghostIndex += 1
    return s




def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Making the Pacman a big mean eating machine!
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    numberOfGhosts = currentGameState.getNumAgents() - 1
    for i in range(numberOfGhosts):
        score += analyzeGhosts(currentGameState, i + 1, pacmanPos)
    score += analyzeFood(currentGameState, pacmanPos)
    score += analyzePellets(currentGameState, pacmanPos)
    score += 40 / (currentGameState.getNumFood() + 1)
    return score

def manhattanDistance(pt1, pt2):
    return abs(pt1[0] - pt2[0]) + abs(pt1[1] - pt2[1])

def analyzeGhosts(gameState, ghostNum, pacmanPos):
    scaredTime = gameState.getGhostState(ghostNum).scaredTimer
    ghostPos = gameState.getGhostPosition(ghostNum)
    distance = manhattanDistance(pacmanPos, ghostPos)
    if scaredTime > 4:
        return 50 / distance
    else:
        return math.log(distance + 1) 

def analyzeFood(gameState, pacmanPos):
    foodGrid = gameState.getFood()
    foodList = foodGrid.asList()
    if not foodList:
        return 0
    closest = 9999
    for food in foodList:
        c = manhattanDistance(pacmanPos, food)
        closest = min(c, closest)
    return 10 / closest

def analyzePellets(gameState, pacmanPos):
    capsules = gameState.getCapsules()
    if not capsules:
        return 0
    closestCapsule = 9999
    for capsule in capsules:
        closestCapsule = min(closestCapsule, manhattanDistance(pacmanPos, capsule))
    return 50 / closestCapsule

def actionsForPacmanExceptStop(gameState):
    actions = []
    for action in gameState.getLegalActions(0):
        if action != Directions.STOP:
            actions.append(action)
    return actions
# Abbreviation
better = betterEvaluationFunction