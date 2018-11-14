from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions, Actions
import game
from util import nearestPoint

#################
# Team creation #
#################

gameTurn = 600


def createTeam(firstIndex, secondIndex, isRed, first='attackAgent', second='attackAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """

    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class attackAgent(CaptureAgent):
    """
    Can use:
    self.myPos(before action), self.start, self.mapHeight, self.mapWidth, self.attacking, self.bestActions, self.legalActions
    self.super, self,superTimeLeft
    """

    def registerInitialState(self, gameState):  # 15s
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)
        # self.foodInStomach = 0
        # self.super = False
        # self.superTimeLeft = 0
        self.mapWidth = gameState.getWalls().width
        self.mapHeight = gameState.getWalls().height
        self.lastAction = None
        if self.red:
            self.safeX = self.mapWidth / 2 - 1
        else:
            self.safeX = self.mapWidth / 2
        self.ghostsDistanceHistory = []
        self.actionHistory = []
        self.escapeAction = None
        self.safeDistance = 0
        self.safePosition = None
        self.updateAgentState(gameState)
        self.possiblePaths = self.findPossiblePaths(gameState)
        self.setState()
        self.deadRoadRecord = {}

    def setState(self):
        self.attack = True

    def updateAgentState(self, gameState):
        self.myState = gameState.getAgentState(self.index)
        self.isPacman = self.myState.isPacman
        self.position = self.myState.getPosition()
        self.enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        self.ghosts = [a for a in self.enemies if not a.isPacman and a.getPosition() is not None and a.scaredTimer == 0]
        self.allghosts = [a for a in self.enemies if not a.isPacman and a.getPosition() is not None]
        self.invaders = [a for a in self.enemies if a.isPacman and a.getPosition() is not None]
        if not self.isPacman:
            self.escapeAction = None
            self.safeDistance = 0
        else:
            self.escapeAction = self.escape(gameState)

        if self.ghosts:
            ghostsDistance = [self.getMazeDistance(a.getPosition(), self.position) for a in self.ghosts]
            self.ghostsDistanceHistory.append(min(ghostsDistance))
        else:
            self.ghostsDistanceHistory.append(9999)

        if self.safePosition:
            if self.position == self.safePosition:
                self.safePosition = None

        # if (self.lastAction):
        # self.updateSuper(gameState, self.lastAction)
        # self.updateFoodInStomach(gameState, self.lastAction)

    def getGhostDistance(self, gameState):
        if self.ghosts and gameState.isPacman:
            ghostDistance = [self.getMazeDistance(a.getPosition(), gameState.getPosition()) for a in self.ghosts]
            return min(ghostDistance)
        else:
            return -9999

    def chooseAction(self, gameState):

        """
        Picks among the actions with the highest Q(s,a).
        """
        # You can profile your evaluation time by uncommenting these lines
        start = time.time()
        action = self.getAttackAction(gameState)

        self.actionHistory.append(action)
        global gameTurn
        gameTurn -= 1

        print time.time() - start
        print self.safePosition

        return action

        # self.updateFoodInStomach(gameState,theAction)

        # print(self.super," ",self.superTimeLeft," ",self.foodInStomach," ",self.isPacman)

        # else:#if defending
        # return self.getDefendAction(gameState)

    def findGhosts(self):

        if len(self.ghostsDistanceHistory) >= 5 and 9999 > self.ghostsDistanceHistory[-1] == self.ghostsDistanceHistory[
            -2] == self.ghostsDistanceHistory[-3] == self.ghostsDistanceHistory[-4] == self.ghostsDistanceHistory[-5] and self.myState.numCarrying>0:
            return True

        if self.ghostsDistanceHistory[-1] < 3:
            return True

        return False

    def getAttackAction(self, gameState):

        self.updateAgentState(gameState)

        enemyFoodLeft = len(self.getFood(gameState).asList())  # getFood returns enemy food
        # myFoodLeft = len(self.getFoodYouAreDefending(gameState).asList())

        if enemyFoodLeft <= 2 and self.isPacman:  # go back home
            return self.escapeAction

        if gameTurn < self.safeDistance * 2 + 5 and self.isPacman:
            return self.escapeAction

        if self.findGhosts() and self.isPacman:
            # print self.myState.numCarrying
            return self.escapeEnemy(gameState)  # escape

        action = self.findFood(gameState)
        if len(self.actionHistory) >= 5 and action == self.actionHistory[-2] == self.actionHistory[-4] and \
                self.actionHistory[-1] == self.actionHistory[-3] == self.actionHistory[
            -5] and action == Actions.reverseDirection(self.actionHistory[-1]) and self.ghostsDistanceHistory[-1] > 2:
            action = random.choice(gameState.getLegalActions(self.index))

        return action

    # def getDefendAction(self,gameState):
    #    return random.choice(self.bestActions)

    def isDeadRoad(self, gameState, position1, position2):
        if (position1, position2) in self.deadRoadRecord:
            return self.deadRoadRecord[(position1, position2)]

        x2, y2 = position1

        from util import PriorityQueue
        openList = PriorityQueue()
        closeList = []
        path = []

        walls = gameState.getWalls().deepCopy()

        walls[int(x2)][int(y2)] = True


        def findPath(node):
            if node[1]:  # If it is not a start state
                findPath(node[1])
                path.append(node[2])

        def heruistic1(positionx):
            x, y = positionx
            return abs(x - self.safeX)

        startPosition = position2
        startNode = (startPosition, [], [], 0)
        openList.push(startNode, heruistic1(startPosition))

        while not openList.isEmpty():
            currentNode = openList.pop()
            currentPosition = currentNode[0]
            if currentPosition not in closeList:
                if self.isHome(currentPosition):
                    findPath(currentNode)
                    for p in closeList:
                        self.deadRoadRecord[(position1, p)] = False
                    self.deadRoadRecord[(position1, position2)] = False
                    return False
                closeList.append(currentPosition)
                for position in Actions.getLegalNeighbors(currentPosition, walls):
                    lastNode = currentNode[1]
                    if lastNode:
                        if not position == currentPosition and not position == lastNode[0]:
                            if (position1, position) in self.deadRoadRecord:
                                for p in closeList:
                                    self.deadRoadRecord[(position1, p)] = self.deadRoadRecord[(position1, position)]
                                return self.deadRoadRecord[(position1, position)]
                            action = Actions.vectorToDirection(
                                (position[0] - currentPosition[0], position[1] - currentPosition[1]))
                            openList.push((position, currentNode, action, currentNode[3] + 1),
                                          currentNode[3] + 1 + heruistic1(position))
                    else:
                        if not position == currentPosition:
                            if (position1, position) in self.deadRoadRecord:
                                for p in closeList:
                                    self.deadRoadRecord[(position1, p)] = self.deadRoadRecord[(position1, position)]
                                return self.deadRoadRecord[(position1, position)]
                            action = Actions.vectorToDirection(
                                (position[0] - currentPosition[0], position[1] - currentPosition[1]))
                            openList.push((position, currentNode, action, currentNode[3] + 1),
                                          currentNode[3] + 1 + heruistic1(position))

            else:
                for p in closeList:
                    self.deadRoadRecord[(position1, p)] = False
                self.deadRoadRecord[(position1, position2)] = False
                return False


        for p in closeList:
            self.deadRoadRecord[(position1, p)] = True
        return True

    def escape(self, gameState):

        from util import PriorityQueue
        openList = PriorityQueue()
        closeList = []
        path = []

        walls = gameState.getWalls().deepCopy()

        for a in self.ghosts:
            x, y = a.getPosition()

            walls[int(x)][int(y)] = True
            # print x1, y1
            if not x + 1 == self.safeX:
                walls[int(x + 1)][int(y)] = True

            walls[int(x)][int(y + 1)] = True
            if not x - 1 == self.safeX:
                walls[int(x - 1)][int(y)] = True

            walls[int(x)][int(y - 1)] = True



        if self.myState.scaredTimer > 0:
            for a in self.invaders:
                x, y = a.getPosition()

                # print x1, y1
                walls[int(x)][int(y)] = True
                walls[int(x + 1)][int(y)] = True
                walls[int(x)][int(y + 1)] = True
                walls[int(x - 1)][int(y)] = True
                walls[int(x)][int(y - 1)] = True

        for a in self.invaders:
            x, y = a.getPosition()
            if x == self.safeX:
                if self.red:
                    walls[int(x + 1)][int(y)] = True
                else:
                    walls[int(x - 1)][int(y)] = True

        def findPath(node):
            if node[1]:  # If it is not a start state
                findPath(node[1])
                path.append(node[2])

        def heruistic2(position1):
            x, y = position1
            return abs(x - self.safeX)

        myState = gameState.getAgentState(self.index)
        startPosition = myState.getPosition()
        startNode = (startPosition, [], [], 0)
        openList.push(startNode, heruistic2(startPosition))

        while not openList.isEmpty():
            currentNode = openList.pop()
            currentPosition = currentNode[0]
            if currentPosition not in closeList:
                if self.isHome(currentPosition):
                    findPath(currentNode)
                    self.safeDistance = len(path)
                    return path[0]
                closeList.append(currentPosition)
                for position in Actions.getLegalNeighbors(currentPosition, walls):
                    action = Actions.vectorToDirection(
                        (position[0] - currentPosition[0], position[1] - currentPosition[1]))
                    openList.push((position, currentNode, action, currentNode[3] + 1),
                                  currentNode[3] + 1 + heruistic2(position))

        self.safeDistance = 0

        if gameState.getLegalActions(self.index):
            legalActions = gameState.getLegalActions(self.index)
            distance = []
            for action in legalActions:
                successor = self.getSuccessor(gameState, action)
                myState = successor.getAgentState(self.index)
                distance.append(self.getGhostDistance(myState))

            print distance
            maxD = max(distance)
            bestActions = [a for a, v in zip(legalActions, distance) if v == maxD]
            return random.choice(bestActions)
        else:
            return Directions.STOP

    def escapeEnemy(self, gameState):

        from util import PriorityQueue
        openList = PriorityQueue()
        closeList = []
        path = []

        walls = gameState.getWalls().deepCopy()

        for a in self.ghosts:
            x, y = a.getPosition()

            # print x1, y1
            walls[int(x)][int(y)] = True
            if not x + 1 == self.safeX:
                walls[int(x + 1)][int(y)] = True

            walls[int(x)][int(y + 1)] = True
            if not x - 1 == self.safeX:
                walls[int(x - 1)][int(y)] = True

            walls[int(x)][int(y - 1)] = True



        if self.myState.scaredTimer > 0:
            for a in self.invaders:
                x, y = a.getPosition()

                # print x1, y1
                walls[int(x)][int(y)] = True
                walls[int(x + 1)][int(y)] = True
                walls[int(x)][int(y + 1)] = True
                walls[int(x - 1)][int(y)] = True
                walls[int(x)][int(y - 1)] = True

        for a in self.invaders:
            x, y = a.getPosition()
            if x == self.safeX:
                if self.red:
                    walls[int(x + 1)][int(y)] = True
                else:
                    walls[int(x - 1)][int(y)] = True



        def findPath(node):
            if node[1]:  # If it is not a start state
                findPath(node[1])
                path.append(node[2])

        def heruistic3(position1):
            x, y = position1
            if self.getCapsules(gameState):
                capsuleD = min([abs(x - c[0]) + abs(y - c[1]) for c in self.getCapsules(gameState)])
                return min(abs(x - self.safeX),capsuleD)
            return abs(x - self.safeX)

        myState = gameState.getAgentState(self.index)
        startPosition = myState.getPosition()
        startNode = (startPosition, [], [], 0)
        openList.push(startNode, heruistic3(startPosition))

        while not openList.isEmpty():
            currentNode = openList.pop()
            currentPosition = currentNode[0]
            if currentPosition not in closeList:
                if self.isSafePosition(currentPosition, gameState):
                    findPath(currentNode)
                    return path[0]
                closeList.append(currentPosition)
                for position in Actions.getLegalNeighbors(currentPosition, walls):
                    action = Actions.vectorToDirection(
                        (position[0] - currentPosition[0], position[1] - currentPosition[1]))
                    openList.push((position, currentNode, action, currentNode[3] + 1),
                                  currentNode[3] + 1 + heruistic3(position))

        return self.escapeAction

    def findFood(self, gameState):

        from util import PriorityQueue
        openList = PriorityQueue()
        closeList = []
        path = []

        walls = gameState.getWalls().deepCopy()

        myTeam = self.getTeam(gameState)
        for each in myTeam:  # assume myTeam only has 2 indices
            if (each != self.index):  friendIndex = each

        a, b = gameState.getAgentState(friendIndex).getPosition()
        if (abs(a - self.safeX) < 3 or gameState.getAgentState(friendIndex).isPacman) and not (a, b) == self.position:
            walls[int(a)][int(b)] = True
            walls[int(a - 1)][int(b)] = True
            walls[int(a + 1)][int(b)] = True
            walls[int(a)][int(b - 1)] = True
            walls[int(a)][int(b + 1)] = True

        for a in self.ghosts:
            x, y = a.getPosition()

            # print x1, y1
            walls[int(x)][int(y)] = True
            if not x + 1 == self.safeX:
                walls[int(x + 1)][int(y)] = True

            walls[int(x)][int(y + 1)] = True
            if not x - 1 == self.safeX:
                walls[int(x - 1)][int(y)] = True

            walls[int(x)][int(y - 1)] = True



        if self.myState.scaredTimer>0:
            for a in self.invaders:
                x, y = a.getPosition()

                # print x1, y1
                walls[int(x)][int(y)] = True
                walls[int(x + 1)][int(y)] = True
                walls[int(x)][int(y + 1)] = True
                walls[int(x - 1)][int(y)] = True
                walls[int(x)][int(y - 1)] = True

        for a in self.invaders:
            x, y = a.getPosition()
            if x == self.safeX:
                if self.red:
                    walls[int(x+1)][int(y)] = True
                else:
                    walls[int(x-1)][int(y)] = True


        def findPath(node):
            if node[1]:  # If it is not a start state
                findPath(node[1])
                path.append(node[2])

        def heruistic4(position1, gameState):
            foodList = self.getFood(gameState).asList()
            if len(foodList)> 0:    #it should not happen
                return min([abs(position1[0] - food[0]) + abs(position1[1] - food[1]) for food in foodList])
            return 9999

        myState = gameState.getAgentState(self.index)
        startPosition = myState.getPosition()
        startNode = (startPosition, [], [], 0)
        openList.push(startNode, heruistic4(startPosition, gameState))

        while not openList.isEmpty():
            currentNode = openList.pop()
            currentPosition = currentNode[0]
            if currentPosition not in closeList:
                if self.isFood(currentPosition, gameState):

                    if currentPosition in self.getCapsules(gameState):  #assume eat capsules can easily get out of hutong
                        if not self.safePosition and self.isDeadRoad(gameState, startPosition, currentPosition):
                            self.safePosition = startPosition
                        findPath(currentNode)
                        return path[0]

                    if self.safePosition:
                        if self.allghosts:
                            ghostsDistance = [max(self.getMazeDistance(a.getPosition(), self.safePosition), a.scaredTimer) for a in
                                              self.allghosts]
                            if self.getMazeDistance(startPosition, currentPosition) + self.getMazeDistance(
                                    currentPosition,
                                    self.safePosition) < min(
                                ghostsDistance):
                                findPath(currentNode)
                                return path[0]
                        else:
                            findPath(currentNode)
                            return path[0]
                    else:
                        if self.isPacman:
                            if self.isDeadRoad(gameState, startPosition, currentPosition):
                                safePosition = startPosition
                                if self.allghosts:
                                    ghostsDistance = [
                                        max(self.getMazeDistance(a.getPosition(), safePosition), a.scaredTimer) for a in
                                        self.allghosts]
                                    if self.getMazeDistance(startPosition, currentPosition) + self.getMazeDistance(
                                            currentPosition,
                                            safePosition) < min(
                                        ghostsDistance):
                                        findPath(currentNode)
                                        self.safePosition = safePosition
                                        return path[0]

                                else:
                                    findPath(currentNode)
                                    self.safePosition = safePosition
                                    return path[0]
                            else:
                                findPath(currentNode)
                                return path[0]
                        else:
                            findPath(currentNode)
                            return path[0]

                closeList.append(currentPosition)
                for position in Actions.getLegalNeighbors(currentPosition, walls):
                    action = Actions.vectorToDirection(
                        (position[0] - currentPosition[0], position[1] - currentPosition[1]))
                    openList.push((position, currentNode, action, currentNode[3] + 1),
                                  currentNode[3] + 1 + heruistic4(position, gameState))

        if self.escapeAction:
            return self.escapeAction
        else:
            if Actions.getLegalNeighbors(startPosition, walls):
                return random.choice([Actions.vectorToDirection(
                        (position[0] - startPosition[0], position[1] - startPosition[1])) for position in Actions.getLegalNeighbors(startPosition, walls)])
            else:
                return Directions.STOP

    def isFood(self, position, gameState):
        x, y = position
        return (self.getFood(gameState)[int(x)][int(y)] or ((x, y) in self.getCapsules(gameState)))

    def isSafePosition(self, position, gameState):
        x, y = position
        if x == self.safeX:
            return True
        else:
            if (x, y) in self.getCapsules(gameState):
                return True
            return False

    def isHome(self, position):
        x, y = position
        if x == self.safeX:
            return True
        else:
            return False

    def findPossiblePaths(self, gameState):  # return nothing

        yMin = 1

        yMax = self.mapHeight - 2

        walls = gameState.getWalls()

        y1 = yMin

        y2 = yMin

        possiblePaths = []

        while (y1 <= yMax):

            while walls[self.safeX][y1]:
                y1 += 1
                if y1 >= self.mapHeight - 1:
                    break

            y2 = y1

            while not walls[self.safeX][y2]:
                y2 += 1
                if y2 >= self.mapHeight:
                    break

            y2 -= 1

            possiblePaths.append((y1, y2))

            y1 = y2 + 1

        return possiblePaths

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
        Returns a counter of features for the state
        """
        if (True):
            # features: successorScore, distanceToFood, distanceToFriend, distanceToMidLine, numOneStepAwayEnemy, distanceToClostEnemy
            features = util.Counter()
            successor = self.getSuccessor(gameState, action)  # successor is a gameState
            features['dangerous'] = self.safeDistance * self.myState.numCarrying
            foodList = self.getFood(successor).asList()
            features['successorScore'] = -len(foodList)  # self.getScore(successor)

            # Compute distance to the nearest food

            if len(foodList) > 0:  # This should always be True
                nextPos = successor.getAgentState(self.index).getPosition()  # myPos here is the position after action
                minDistance = min([self.getMazeDistance(nextPos, food) for food in foodList])
                features['distanceToFood'] = minDistance

            """
            my code here:
            """
            # eating food: weight: 10
            if (nextPos in foodList):
                features['eatingFood'] = 1
            else:
                features['eatingFood'] = 0

            # (use)distance to teammate agent (assume teammate is still): weight 0.5

            myTeam = self.getTeam(gameState)
            for each in myTeam:  # assume myTeam only has 2 indices
                if (each != self.index):  friendIndex = each

            nextGameState = gameState.generateSuccessor(self.index, action)

            friendPosition = gameState.getAgentPosition(friendIndex)
            myPosition = gameState.getAgentPosition(self.index)

            friendNextPosition = nextGameState.getAgentPosition(friendIndex)  # is this always same with friendPosition?

            friendDistance = self.getMazeDistance(nextPos, friendNextPosition)
            features["distanceToFriend"] = friendDistance

            # (use)manhattan distance to the midline: weight -0.1
            mapHeight = gameState.getWalls().height  # useful
            mapWidth = gameState.getWalls().width  # useful
            midlineX = (mapWidth - 1) / 2
            myX = nextPos[0]
            features["distanceToMidLine"] = abs(myX - midlineX)

            # attack: number of active ghost positions list one step away : weight -10

            numberOneStepAway = 0
            distanceToClosestGhost = 1000

            for ghost in self.ghosts:
                dis = self.getMazeDistance(nextPos, ghost.getPosition())
                if dis == 1:
                    numberOneStepAway += 1
                if dis < distanceToClosestGhost:
                    distanceToClosestGhost = dis

            features["numOneStepAwayEnemy"] = numberOneStepAway

            # attack: distance to closest ghost:  # there should be always a value returned? weight: 1
            # if (distanceToClostGhost < 1000): #1
            features["distanceToClostEnemy"] = distanceToClosestGhost
            # else:
            #    features["distanceToClostEnemy"] = 0

            return features

    def getWeights(self, gameState, action):

        return {'successorScore': 100, 'distanceToFood': -1, 'distanceToFriend': 0,
                'distanceToMidLine': -0.2, 'numOneStepAwayEnemy': -10, 'distanceToClostEnemy': 0.5,
                'eatingFood': 2, 'dangerous': 0}

    def evaluateD(self, gameState, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.getDFeatures(gameState, action)
        weights = self.getDWeights(gameState, action)
        return features * weights

    def getDefendAction(self, gameState):

        self.updateAgentState(gameState)

        enemyFoodLeft = len(self.getFood(gameState).asList())  # getFood returns enemy food
        # myFoodLeft = len(self.getFoodYouAreDefending(gameState).asList())

        legalActions = gameState.getLegalActions(self.index)
        values = [self.evaluateD(gameState, a) for a in legalActions]
        maxValue = max(values)
        bestActions = [a for a, v in zip(legalActions, values) if v == maxValue]

        return random.choice(bestActions)

    def getDFeatures(self, gameState, action):

        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        nextPos = myState.getPosition()
        x = int(nextPos[0])
        y = int(nextPos[1])
        nextPos = (x, y)

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)

        minDis = 1000
        if len(invaders) > 0:
            dists = [self.getMazeDistance(nextPos, a.getPosition()) for a in invaders]
            minDis = min(dists)
            features['invaderDistance'] = minDis

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        # self.enemyIndexList = self.getOpponents(gameState)
        # for eachIndex in self.enemyIndexList
        if gameState.getAgentState(self.index).scaredTimer > minDis / 2 and len(invaders) > 0:
            features['distanceToSuperPacman'] = minDis

        self.possiblePathsMidNodes = []
        for each in self.possiblePaths:
            self.possiblePathsMidNodes.append((self.safeX, int((each[0] + each[1]) / 2)))

        minDis = 1000
        enemyObserable = False
        for eachEnemy in self.getOpponents(successor):
            eachEnemyPos = successor.getAgentPosition(eachEnemy)  # may be None
            if eachEnemyPos is None:
                continue
            else:
                enemyObserable = True
            for eachPathNode in self.possiblePathsMidNodes:
                theDis = self.getMazeDistance(eachEnemyPos, eachPathNode)
                if theDis < minDis:
                    minDis = theDis
                    defendPathNode = eachPathNode

        if enemyObserable:
            features['destanceToDoor'] = self.getMazeDistance(defendPathNode, nextPos)
            print(defendPathNode)
            print(nextPos)
            # print(features['destanceToDoor'])

        # features['distanceToMidNode'] = self.getMazeDistance(nextPos, (self.safeX, int(self.mapHeight / 2)))

        return features

    def getDWeights(self, gameState, action):
        return {'onDefense': 100000, 'numInvaders': -5000, 'invaderDistance': -1000, 'stop': -100, 'reverse': -2,
                'distanceToSuperPacman': 2000, 'destanceToDoor': -200}


class defendAgent(attackAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def setState(self):
        self.attack = False
