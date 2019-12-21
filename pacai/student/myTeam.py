from pacai.agents.capture.reflex import ReflexCaptureAgent
from pacai.util import counter
import random
from pacai.core.directions import Directions
from pacai.core.actions import Actions


def createTeam(firstIndex, secondIndex, isRed,
               first='pacai.student.myTeam.offense',
               second='pacai.student.myTeam.DefensiveAgent'):
    """
    ctong3; hwang108
    This function should return a list of two agents that will form the capture team,
    initialized using firstIndex and secondIndex as their agent indexed.
    isRed is True if the red team is being created,
    and will be False if the blue team is being created.
    """

    firstAgent = offense
    secondAgent = DefensiveAgent

    return [
        firstAgent(firstIndex),
        secondAgent(secondIndex),
    ]


def closestFood(pos, food, walls):
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        x, y, distance = fringe.pop(0)
        if (x, y) in expanded:
            continue
        expanded.add((x, y))
        if food[x][y]:
            return distance
        neighbours = Actions.getLegalNeighbors((x, y), walls)
        for neighbours_x, neighbours_y in neighbours:
            fringe.append((neighbours_x, neighbours_y, distance + 1))
    return None


class offense(ReflexCaptureAgent):

    def __init__(self, index, **kwargs):
        super().__init__(index)

    #  use weights times feature to get the qvalue
    def getQValue(self, gameState, action):
        feature = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        qvalue = feature * weights
        return qvalue

    # loop through legal actions to get qvalues for each action
    # then return the max one
    def getValue(self, gameState):
        qValues = []
        legalActions = gameState.getLegalActions(self.index)
        if len(legalActions) == 0:
            return 0.0
        else:
            for action in legalActions:
                qValues.append(self.getQValue(gameState, action))
            return max(qValues)

    # return the best action
    def getPolicy(self, gameState):
        value = self.getValue(gameState)
        actions = []
        legalActions = gameState.getLegalActions(self.index)
        legalActions.remove(Directions.STOP)
        append = False
        for action in legalActions:
            if self.getQValue(gameState, action) == value or len(legalActions) == 1:
                actions.append(action)
                append = True
        if not append:
            actions.append(random.choice(legalActions))
        if len(actions) == 0:
            return None
        else:
            return random.choice(actions)

    def chooseAction(self, gameState):
        action = self.getPolicy(gameState)
        return action

    def getWeights(self, gameState, action):
        ghostStateList = []
        pacmanStateList = []
        enemyList = self.getOpponents(gameState)
        for enemy in enemyList:
            ghostState = gameState.getAgentState(enemy)
            if not ghostState.isPacman():
                ghostStateList.append(ghostState)
            else:
                pacmanStateList.append(ghostState)
        scaredGhost = False
        for ghosts in ghostStateList:
            if ghosts.isScared():
                scaredGhost = True
        if scaredGhost:
            return {'successorScore': -1,
                    'ghost-near-by': 0,
                    'closest-food': -2,
                    'distToEnemy': 0,
                    'eats-food': 10,
                    'eat_capsules': 10
                    }
        else:
            ghostCoordinate = []
            for things in ghostStateList:
                ghostCoordinate.append(things.getPosition())
            currentPos = gameState.getAgentPosition(self.index)
            distanceList = [self.getMazeDistance(currentPos, i) for i in ghostCoordinate]
            if len(distanceList) > 0:
                minDistance = min(distanceList)
                # if ghost is coming, try to escape rather than eat food
                if minDistance <= 3:
                    return {'successorScore': -1,
                            'ghost-near-by': -0.2,
                            'closest-food': -2,
                            'distToEnemy': -2,
                            'eats-food': 10,
                            'eat_capsules': 10
                            }

                return {'successorScore': -1,
                        'ghost-near-by': -0.2,
                        'closest-food': -2.5,
                        'distToEnemy': -1,
                        'eats-food': 10,
                        'eat_capsules': 10
                        }
            else:
                return {'successorScore': -1,
                        'ghost-near-by': -0.2,
                        'closest-food': -2,
                        'distToEnemy': -1,
                        'eats-food': 10,
                        'eat_capsules': 10
                        }

    def getGhost(self, gameState):
        ghostList = []
        enemies = self.getOpponents(gameState)
        for enemy in enemies:
            temp = gameState.getAgentState(enemy)
            if not temp.isPacman():
                pos = gameState.getAgentPosition(enemy)
                ghostList.append(pos)

        return ghostList

    def getFeatures(self, gameState, action):
        Team = self.getTeam(gameState)
        if Team == [1, 3]:
            food = gameState.getRedFood()
            capsules = gameState.getRedCapsules()
        else:
            food = gameState.getBlueFood()
            capsules = gameState.getBlueCapsules()
        # get walls and ghosts
        walls = gameState.getWalls()
        ghosts = self.getGhost(gameState)

        # use code from featureExtractors.py as reference

        features = counter.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        x, y = gameState.getAgentPosition(self.index)
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        features["ghost-near-by"] = sum(
            (next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)
        if not features["ghost-near-by"] and food[next_x][next_y]:
            features["eats-food"] = 1.0

        dist = closestFood((next_x, next_y), food, walls)
        if dist is not None:
            # print(walls.getWidth())
            features["closest-food"] = float(dist) / (walls.getWidth() * walls.getHeight())

        if len(ghosts) > 0:
            disToEachPacman = [
                self.getMazeDistance((x, y), a) for a in ghosts
                if a is not None]
            shortestDist = min(disToEachPacman)
            features['distToEnemy'] = shortestDist

        if (next_x, next_y) in capsules:
            # print("next is capsules")
            features['eat_capsules'] = 1

        features.divideAll(10.0)

        return features


class DefensiveAgent(ReflexCaptureAgent):

    def __init__(self, index, **kwargs):
        super().__init__(index)

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the agent and populates useful fields,
        such as the team the agent is on and the `pacai.core.distanceCalculator.Distancer`.

        IMPORTANT: If this method runs for more than 15 seconds, your agent will time out.
        """
        # returns corrdinate of opponents
        self.opponents = self.getOpponents(gameState)

        self.theirFoodList = self.getFood(gameState)
        self.ourFoodList = self.getFoodYouAreDefending(gameState)
        self.theirCapsuleList = self.getCapsules(gameState)
        self.ourCapsuleList = self.getCapsulesYouAreDefending(gameState)
        self.defensiveMode = True

        super().registerInitialState(gameState)

        # Your initialization code goes here, if you need any.

    def chooseAction(self, gameState):

        actions = gameState.getLegalActions(self.index)
        pos = gameState.getAgentPosition(self.index)

        enemiesAsPacman = []

        result = Directions.STOP

        enemies = []
        for action in actions:
            successor = self.getSuccessor(gameState, action)
            # myState = successor.getAgentState(self.index)
            enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]

        # To see if there's any enemy that turns into Pacman already
        # If they are, add them into the list
        for enemy in enemies:
            if enemy.isPacman() is True:
                self.defensiveMode = True
                enemiesAsPacman.append(enemy)
            else:
                self.defensiveMode = False
                startingPoint = (0, 0)
                layoutWidth = gameState.getInitialLayout().getWidth()
                layoutHeight = gameState.getInitialLayout().getHeight()
                boundaryWidth = int(layoutWidth / 2)
                boundaryHeight = int(layoutHeight / 2)

                # If team is on blue team, which is right side of the map
                if gameState.isOnRedTeam(self.index):
                    boundaryWidth -= 2
                walls = gameState.getWalls()
                # while is wall
                while walls[boundaryWidth][boundaryHeight]:
                    # move up
                    boundaryHeight += 1
                    if boundaryHeight > layoutHeight:
                        boundaryHeight = int(layoutHeight / 2)
                        break
                while walls[boundaryWidth][boundaryHeight]:
                    # move down
                    boundaryHeight -= 1
                startingPoint = (boundaryWidth, boundaryHeight)
                result = self.BFS(pos, startingPoint, gameState)

        if len(enemiesAsPacman) > 0:
            for pacmans in enemiesAsPacman:
                pacmanPosition = pacmans.getPosition()
                result = self.BFS(pos, pacmanPosition, gameState)
        return result

    # Not BFS :) just return the first action of the shortest maze distance
    def BFS(self, startPos, goalPos, gameState):
        actions = gameState.getLegalActions(self.index)
        successorList = []
        for action in actions:
            if action == Directions.STOP:
                continue
            successor = gameState.generateSuccessor(self.index, action)
            oldCoordinate = successor.getAgentState(self.index).getPosition()
            coordinate = (int(oldCoordinate[0]), int(oldCoordinate[1]))
            successorList.append((coordinate, action))

        bestDistance = 100000
        bestAction = Directions.STOP
        for connectingSuccessors in successorList:
            currentDistance = self.getMazeDistance(connectingSuccessors[0], goalPos)
            if currentDistance < bestDistance:
                bestDistance = currentDistance
                bestAction = connectingSuccessors[1]
        return bestAction
