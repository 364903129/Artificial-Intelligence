import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent


def manhattanDistance(xy1, xy2):
    """Returns the Manhattan distance between points xy1 and xy2"""
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])


class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        # print(bestScore)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.
        # print(chosenIndex)

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        # newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()

        # *** Your Code Here ***
        newPosition = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        win = successorGameState.isWin()
        lose = successorGameState.isLose()
        step = 4
        if win:
            return 5000
        if lose:
            return -5000
        currentscore = successorGameState.getScore()
        hideEnemy = lambda enemy: manhattanDistance(newPosition, enemy.getPosition())
        minManhattanGhost = min(map(hideEnemy, newGhostStates))
        currentscore = currentscore - 10 * (step - min(minManhattanGhost, step))
        getFood = lambda food: manhattanDistance(newPosition, food)
        minManhattanFood = min(map(getFood, newFood.asList()))
        currentscore = currentscore + 10 / minManhattanFood

        return currentscore


class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def getAction(self, gameState):

        return self.MinimaxSearch(gameState, 1, 0)

    def MinimaxSearch(self, gameState, depth, agentIndex):
        if gameState.isWin() or gameState.isLose() or depth > self.getTreeDepth():
            return self.getEvaluationFunction()(gameState)

        legal_moves = []
        for move in gameState.getLegalActions(agentIndex):
            if move != 'Stop':
                legal_moves.append(move)

        index = agentIndex + 1
        if index >= gameState.getNumAgents():
            depth += 1
            index = 0

        results = [self.MinimaxSearch(gameState.generateSuccessor(agentIndex, action),
                                      depth, index) for action in legal_moves]
        bestInd = []
        if agentIndex == 0 and depth == 1:
            path = max(results)
            for index in range(len(results)):
                if results[index] == path:
                    bestInd.append(index)
            chosenIndex = random.choice(bestInd)
            return legal_moves[chosenIndex]

        if agentIndex == 0:
            path = max(results)
            return path
        else:
            path = min(results)
            return path


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def getAction(self, gameState):
        action = self.AlphaBeta(gameState, 1, 0, -9999999999, 9999999999)
        return action

    def AlphaBeta(self, gameState, depth, agent, A, B):
        if depth > self.getTreeDepth() or gameState.isWin() or gameState.isLose():
            return self.getEvaluationFunction()(gameState)

        legal_moves = []
        for move in gameState.getLegalActions(agent):
            if move != 'Stop':
                legal_moves.append(move)

        index = agent + 1
        if index >= gameState.getNumAgents():
            depth += 1
            index = 0

        if agent == 0 and depth == 1:
            results = [
                self.AlphaBeta(gameState.generateSuccessor(agent, action),
                               depth, index, A, B) for
                action in legal_moves]
            path = max(results)
            bestInd = []
            for index in range(len(results)):
                if results[index] == path:
                    bestInd.append(index)
            chosenIndex = random.choice(bestInd)
            return legal_moves[chosenIndex]

        if agent == 0:
            path = -9999999999
            for action in legal_moves:
                path = max(path,
                           self.AlphaBeta(gameState.generateSuccessor
                                          (agent, action), depth, index,
                                          A, B))
                if path >= B:
                    return path
                A = max(A, path)
            return path
        else:
            path = 9999999999
            for action in legal_moves:
                path = min(path,
                           self.AlphaBeta(gameState.generateSuccessor(agent, action),
                                          depth, index,
                                          A, B))
                if A >= path:
                    return path
                B = min(B, path)
            return path


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def getAction(self, gameState):
        return self.ExpectiMax(gameState, 1, 0)

    def ExpectiMax(self, gameState, Depth, agent):
        if Depth > self.getTreeDepth() or gameState.isWin() or gameState.isLose():
            return self.getEvaluationFunction()(gameState)

        legal_moves = []
        for move in gameState.getLegalActions(agent):
            if move != 'Stop':
                legal_moves.append(move)

        index = agent + 1
        if index >= gameState.getNumAgents():
            Depth += 1
            index = 0

        results = [self.ExpectiMax(gameState.generateSuccessor(agent, action),
                                   Depth, index) for action in
                   legal_moves]

        if agent == 0 and Depth == 1:
            path = max(results)
            bestInd = []
            for index in range(len(results)):
                if results[index] == path:
                    bestInd.append(index)
            chosenIndex = random.choice(bestInd)
            return legal_moves[chosenIndex]

        if agent == 0:
            path = max(results)
            return path
        else:
            path = sum(results) / len(results)
            return path


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """

    return currentGameState.getScore()


class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index)
