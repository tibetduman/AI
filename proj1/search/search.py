# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    return DFSHelper(problem, [], [], problem.getStartState(), None)

def DFSHelper(problem, pathSoFar, discovered, state, move):
    disc = discovered.copy()
    path = pathSoFar.copy()
    disc.append(state)
    if (move):
        path.append(move)
    if (problem.isGoalState(state)):
        return path
    for successorTriple in problem.getSuccessors(state):
        if (successorTriple[0] not in discovered):
            answer = DFSHelper(problem, path, disc, successorTriple[0], successorTriple[1])
            if (answer):
                return answer
    return []
        

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    sys.setrecursionlimit(20000)
    q = util.Queue()
    q.push([problem.getStartState(), []])
    return BFSHelper(problem, [problem.getStartState()], q)

def BFSHelper(problem, discovered, que):
    disc = discovered.copy()
    if que.isEmpty():
        return
    nextNode = que.pop()
    state = nextNode[0]
    pathSoFar = nextNode[1]
    if (problem.isGoalState(state)):
        return pathSoFar
    for successorTriple in problem.getSuccessors(state):
        s = successorTriple[0]
        if (successorTriple[0] not in disc):
            disc.append(s)
            path = pathSoFar.copy()
            path.append(successorTriple[1])
            que.push([s, path])
    return BFSHelper(problem, disc, que)

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    pq = PriorityQueWithPath()
    pq.push(problem.getStartState(), problem.getCostOfActions([]), [])
    return UCSHelper(problem, [problem.getStartState()], pq, nullHeuristic)

def UCSHelper(problem, discovered, pq, heuristic):
    disc = discovered.copy()
    if pq.isEmpty():
        return
    state, pathSoFar, costSoFar = pq.pop()
    disc.append(state)
    if (problem.isGoalState(state)):
        return pathSoFar
    for successorTriple in problem.getSuccessors(state):
        s = successorTriple[0]
        if (s not in disc):
            path = pathSoFar.copy()
            move = successorTriple[1]
            path.append(move)
            pq.push(s, problem.getCostOfActions(path) + heuristic(s, problem), path)
    return UCSHelper(problem, disc, pq, heuristic)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    sys.setrecursionlimit(20000)
    pq = PriorityQueWithPath()
    pq.push(problem.getStartState(), problem.getCostOfActions([]), [])
    return UCSHelper(problem, [problem.getStartState()], pq, heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

class PriorityQueWithPath:
    """ Implements a priority que structure by making use of the PriorityQue class
    in utils.py, but also stores the path and the cost to the given item alongside it."""
    def __init__(self):
        self.pq = util.PriorityQueue()
        self.dict = {}
        self.costs = {}
    
    def push(self, item, priority, path):
        if item in self.dict:
            self.update(item, priority, path)
            return
        self.pq.push(item, priority)
        self.dict[item] = path
        self.costs[item] = priority
    
    def pop(self):
        item = self.pq.pop()
        path = self.dict.pop(item)
        cost = self.costs.pop(item)
        return item, path, cost
    
    def isEmpty(self):
        return self.pq.isEmpty()

    def update(self, item, priority, path):
        if priority < self.costs[item]:
            self.pq.update(item, priority)
            self.dict[item] = path
            self.costs[item] = priority

