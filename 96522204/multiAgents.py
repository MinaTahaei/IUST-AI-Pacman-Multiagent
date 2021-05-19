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

from game import Agent

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

        "*** YOUR CODE HERE ***"
        
        score = successorGameState.getScore()

        ghostDistance = float('Inf')
        
        for Ghost in newGhostStates:
            
            ghost = util.manhattanDistance(newPos, Ghost.getPosition())
            if ghost < ghostDistance:
                ghostDistance = ghost

        if ghostDistance < 2:
            return float('-Inf')

        foodDistance = float('Inf')
        
        for Food in newFood.asList():
            
            food = util.manhattanDistance(newPos, Food)
            if food < foodDistance:
                foodDistance = food

        score += ghostDistance/foodDistance

        return score


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
        "*** YOUR CODE HERE ***"
        
        def maxValue(gameState, depth):
            
            if gameState.isWin() or gameState.isLose() or depth == 0:
                
                return self.evaluationFunction(gameState)

            value = float('-Inf')
            
            for action in gameState.getLegalActions(0):
                
                value = max(value, minValue(gameState.generateSuccessor(0, action), depth, 1))

            return value

        def minValue(gameState, depth, agents):
            
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)

            value = float('Inf')
            
            for action in gameState.getLegalActions(agents):
                
                if agents == gameState.getNumAgents() - 1:
                    value = min(value, maxValue(gameState.generateSuccessor(agents, action), depth - 1))
                    
                else:
                    value = min(value, minValue(gameState.generateSuccessor(agents, action), depth, agents + 1))

            return value

        def minimaxDecision(gameState):
            maxValue = float('-Inf')
            for action in gameState.getLegalActions(0):
                
                value = minValue(gameState.generateSuccessor(0, action), self.depth, 1)
                
                if value > maxValue:
                    maxValue = value
                    move = action

            return move

        return minimaxDecision(gameState)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        
        def maxValue(gameState, depth, alpha, beta):
            
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState), ""

            value = (float('-Inf'), "")
            for action in gameState.getLegalActions(0):
                Move = minValue(gameState.generateSuccessor(0, action), depth, 1, alpha, beta)

                if Move[0] > value[0]: 
                    value = (Move[0], action)

                if value[0] > beta:
                    return value
                
                alpha = max(value[0], alpha)

            return value

        def minValue(gameState, depth,agents, alpha, beta):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), ""

            value = (float('Inf'), "")
            
            for action in gameState.getLegalActions(agents):
                if agents == gameState.getNumAgents() - 1:
                    Move = maxValue(gameState.generateSuccessor(agents, action), depth + 1, alpha, beta)
                else:
                    Move = minValue(gameState.generateSuccessor(agents, action), depth, agents + 1, alpha, beta)

                if Move[0] < value[0]:
                    value = (Move[0], action)

                if value[0] < alpha:
                    return value
                beta = min(value[0], beta)

            return value

        result = maxValue(gameState, 0, float('-Inf'), float('Inf'))

        return result[1]

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
        "*** YOUR CODE HERE ***"
        
        def expectimax(gameState, agent, depth=0):
            
            legalAction = gameState.getLegalActions(agent)
            agentNumber = gameState.getNumAgents() - 1          
            Action = None

            if (gameState.isLose() or gameState.isWin() or depth == self.depth):
                return [self.evaluationFunction(gameState)]
            
            elif agent == agentNumber:
                depth += 1
                childAgent = self.index
                
            else:
                childAgent = agent + 1

            actionLength = len(legalAction)

            if agent == self.index:
                value = -float("inf")

            else:
                value = 0

            for legalAction in legalAction:
                
                successorGameState = gameState.generateSuccessor(agent, legalAction)
                
                expectedMax = expectimax(successorGameState, childAgent, depth)[0]
                if agent == self.index:
                    if expectedMax > value: 
                        value = expectedMax
                        Action = legalAction
                else:

                    value = value + ((1.0/actionLength) * expectedMax)
                    
            return value, Action

        bestScoreItems = expectimax(gameState, self.index)
        bestScore = bestScoreItems[0]
        Move =  bestScoreItems[1]
        return Move


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Calculating Manhattan distances and evaluating based on the results
    """
    "*** YOUR CODE HERE ***"
    
    currentPosition = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    currentScore = currentGameState.getScore()
    foods = currentFood.asList()


    for food in foods:
        foodDistance = manhattanDistance(currentPosition, food)
        currentScore += (1 / float(foodDistance))

    for ghost in ghostStates:
        
        ghostPosition = ghost.getPosition()
        
        distance = manhattanDistance(currentPosition, ghostPosition)
        
        if distance == 0:
            continue
        if(distance < 3):
            currentScore += 5 * (1 / float(distance))  
        else:
            currentScore += (1 / float(distance)) 

    return currentScore

# Abbreviation
better = betterEvaluationFunction
