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


from util import manhattanDistance
from game import Directions
import random, util
import numpy as np
from game import Agent
import sys
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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



    def evaluationFunction(self, currentGameState, action):
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
        newGhostPositions =successorGameState.getGhostPositions()
        # print('ghostpos',newGhostPositions)

        #print ('newFood',type(newFood),'newPos',newPos,'newGhostStates',newGhostStates,'newScaredTImes',newScaredTimes)
        foodWidth=newFood.width
        foodHeight=newFood.height
        foodDistances=[]
        for x in range(foodWidth):
            for y in range(foodHeight):
                if newFood[x][y]:
                    point2=(x,y)
                    distance2food=manhattanDistance(newPos, point2)
                    foodDistances.append(distance2food)
        # print('foodDistances',foodDistances)
        ghostDistances=[]
        for i in range(len(newGhostPositions)):
            ghostDistances.append(manhattanDistance(newPos,newGhostPositions[i]))
        # print('ghostDistances',ghostDistances)

        alpha=.8
        if sum(foodDistances)==0:
            eval_value=100000000000000  #alpha*(1/.10)+(1-alpha)*sum(ghostDistances)
        elif sum(ghostDistances)==0:
            eval_value=-100000000000000  #((1/(sum(foodDistances)*.1)))*200
        else:
            # eval_value=-alpha*((sum(foodDistances)*.10))+(1-alpha)*sum(ghostDistances) # this worked
           eval_value=(alpha*(1/(sum(foodDistances)*.1))-(1-alpha)*1/sum(ghostDistances))*200
        # print('eval_value',eval_value)
        # print('score',successorGameState.getScore())

        return  eval_value+ successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
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
        self.successor_count=0
        self.iterationcount=0

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):


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
        

   


        self.successor_count=0


        def max_value(state, depth):
            successor_count=self.successor_count
            if state.isWin() or state.isLose() or depth==self.depth:
                return self.evaluationFunction(state)
            actions = state.getLegalActions(0)
            v = -np.inf
            first_min_agent=1       
            for action in actions:
                old_value=v
                v =max(v, min_value(state.generateSuccessor(0, action), depth, first_min_agent))
                self.successor_count+=1
                # print('successor count',self.successor_count)
                if v>old_value:
                    optimal_play= action
            if depth == 0:
                return optimal_play
            else:
                return v

        def min_value(state, depth, min_depth):

        
            if state.isLose() or state.isWin():
                return state.getScore()
          

            actions = state.getLegalActions(min_depth)
            v = np.inf
            # score = v
            for action in actions:
                if min_depth == state.getNumAgents()-1: # We are on the last ghost and it will be Pacman's turn next.
                    old_value=v
                    v = min(old_value,max_value(state.generateSuccessor(min_depth, action), depth + 1))
                    self.successor_count+=1
                else:
                    old_value=v
                    v =min(old_value, min_value(state.generateSuccessor(min_depth, action), depth, min_depth+1))
                    self.successor_count+=1
                    # print('successor count',self.successor_count)
                # if score < v:
                #     v = score
            return v
        bestaction=max_value(gameState, 0)
        # print('successor count 1 iter',self.successor_count)
        return bestaction





           

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """



        def max_value(state, depth,alpha,beta):
            successor_count=self.successor_count
            if state.isWin() or state.isLose() or depth==self.depth:
                return self.evaluationFunction(state)
            actions = state.getLegalActions(0)
            v = -np.inf
            first_min_agent=1       
            for action in actions:
                old_value=v
                # print('v',v,'min_value',min_value(state.generateSuccessor(0, action), depth, first_min_agent,alpha,beta))
                v =max(v, min_value(state.generateSuccessor(0, action), depth, first_min_agent,alpha,beta))
                if v>beta:
                    return v
                # print('Alpha and v',alpha,v)
                alpha=max(alpha,v)
                self.successor_count+=1
                # print('successor count',self.successor_count)
                if v>old_value:
                    optimal_play= action

            if depth == 0:
                return optimal_play
            else:
                return v

        def min_value(state, depth, min_depth,alpha,beta):

        
            if state.isLose() or state.isWin():
                return state.getScore()
          


            actions = state.getLegalActions(min_depth)
            v = np.inf
            # score = v
            for action in actions:
                if min_depth == state.getNumAgents()-1: # We are on the last ghost and it will be Pacman's turn next.
                    old_value=v
                    v = min(old_value,max_value(state.generateSuccessor(min_depth, action), depth + 1,alpha,beta))
                    if v<alpha:
                        return v
                    beta=min(beta,v)
                    self.successor_count+=1
                else:
                    old_value=v
                    v =min(old_value, min_value(state.generateSuccessor(min_depth, action), depth, min_depth+1,alpha,beta))
                    if v<alpha:
                        return v
                    beta=min(beta,v)
                    self.successor_count+=1
                    # print('successor count',self.successor_count)
                # if score < v:
                #     v = score
            return v
        alpha=-np.inf
        beta=np.inf
        bestaction=max_value(gameState, 0,alpha,beta)
        # print('successor count 1 iter',self.successor_count)
        return bestaction



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """

        def max_value(state, depth):
            successor_count=self.successor_count
            if state.isWin() or state.isLose() or depth==self.depth:
                return self.evaluationFunction(state)
            actions = state.getLegalActions(0)
            v = -np.inf
            first_min_agent=1       
            for action in actions:
                old_value=v
                v =max(v, min_value(state.generateSuccessor(0, action), depth, first_min_agent))
                self.successor_count+=1
                # print('successor count',self.successor_count)
                if v>old_value:
                    optimal_play= action
            if depth == 0:
                return optimal_play
            else:
                return v

        def min_value(state, depth, min_depth):

        
            if state.isLose() or state.isWin():
                return state.getScore()
          
 

            actions = state.getLegalActions(min_depth)
            v = 0
            p=1/len(actions)
            # score = v
            for action in actions:
                if min_depth == state.getNumAgents()-1: # We are on the last ghost and it will be Pacman's turn next.
                    old_value=v
                    new_value=max_value(state.generateSuccessor(min_depth, action), depth + 1)
                    v+=new_value*p
                    self.successor_count+=1
                else:
                    old_value=v
                    new_value= min_value(state.generateSuccessor(min_depth, action), depth, min_depth+1)
                    v+=new_value*p
                    self.successor_count+=1
                    # print('successor count',self.successor_count)
                # if score < v:
                #     v = score
            return v
        bestaction=max_value(gameState, 0)
        # print('successor count 1 iter',self.successor_count)
        return bestaction




def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """


 # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newGhostPositions =successorGameState.getGhostPositions()
    # print(sum(newScaredTimes))
    # print('ghostpos',newGhostPositions)

    #print ('newFood',type(newFood),'newPos',newPos,'newGhostStates',newGhostStates,'newScaredTImes',newScaredTimes)
    foodWidth=newFood.width
    foodHeight=newFood.height
    foodDistances=[]
    food_count=0
    for x in range(foodWidth):
        for y in range(foodHeight):
            if newFood[x][y]:
                food_count+=1
                point2=(x,y)
                distance2food=manhattanDistance(newPos, point2)
                foodDistances.append(distance2food)
    # print('foodDistances',foodDistances)
    ghostDistances=[]
    for i in range(len(newGhostPositions)):
        ghostDistances.append(manhattanDistance(newPos,newGhostPositions[i]))
    # print('ghostDistances',ghostDistances)

    alpha=.5
    if food_count==0:
        eval_value=np.inf #alpha*(1/.10)+(1-alpha)*sum(ghostDistances)
    elif sum(ghostDistances)==0:
        eval_value=-100000000000000  #((1/(sum(foodDistances)*.1)))*200
    else:
        # eval_value=-alpha*((sum(foodDistances)*.10))+(1-alpha)*sum(ghostDistances) # this worked
       eval_value=(alpha*(1/(sum(foodDistances)*.1))-(1-alpha)*1/sum(ghostDistances))*2 +1/(.1*food_count)**2 +sum(newScaredTimes)
    # print('eval_value',eval_value)
    # print('score',10*successorGameState.getScore())

    return  eval_value+ successorGameState.getScore()


# Abbreviation
better = betterEvaluationFunction
