# myTeam.py
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions, Actions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'AttackAgent', second = 'AttackAgent'):
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

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class AttackAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    self.weights = util.Counter()
    self.weights["bias"]=195
    self.weights["#-of-ghosts-1-step-away"]=-118
    self.weights["eats-food"]=272
    self.weights["closest-food"]=-5
    self.mapWidth = gameState.getWalls().width


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''

    myState = gameState.getAgentState(self.index)
    startPosition = myState.getPosition()
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    ghosts = []
    if len(invaders) > 0:
      ghosts = [self.getMazeDistance(a.getPosition(), startPosition) for a in invaders]

    ghosts.append(100)

    if min(ghosts)<3 and myState.isPacman:
      return self.escape(gameState)

    return self.getPolicy(gameState)


  def escape(self, gameState):

      from util import PriorityQueue
      openList = PriorityQueue()
      closeList = []
      path = []

      walls = gameState.getWalls()

      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
      if len(invaders) > 0:
        for a in invaders:
          x,y = a.getPosition()
          walls[int(x)][int(y)]=True

      def findPath(node):
        if node[1]:  # If it is not a start state
          findPath(node[1])
          path.append(node[2])

      def heruistic(position1):
        x, y = position1
        return abs(x-(self.mapWidth-1)/2)

      myState = gameState.getAgentState(self.index)
      startPosition = myState.getPosition()
      startNode = (startPosition, [], [], 0)
      openList.push(startNode, heruistic(startPosition))

      while not openList.isEmpty():
        currentNode = openList.pop()
        currentPosition = currentNode[0]
        if currentPosition not in closeList:
          if self.isSafePosition(currentPosition):
            findPath(currentNode)
            return path[0]
          closeList.append(currentPosition)
          for position in Actions.getLegalNeighbors(currentPosition, walls):
            action = Actions.vectorToDirection((position[0]-currentPosition[0],position[1]-currentPosition[1]))
            openList.push((position, currentNode, action, currentNode[3]+1), currentNode[3]+1+heruistic(position))

      return path[0]




  def isSafePosition(self, position):
    x,y = position
    if(self.red):
      if x<(self.mapWidth-1)/2:
        return True
      else:
        return False
    else:
      if x<(self.mapWidth-1)/2:
        return False
      else:
        return True

  def computeValueFromQValues(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    actions = state.getLegalActions(self.index)
    if not actions:
      return 0.0

    value = self.getQValue(state, actions[0])

    for action in actions:
      value = max(value, self.getQValue(state, action))

    return value

  def computeActionFromQValues(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    actions = state.getLegalActions(self.index)

    if not actions:
      return None

    value = self.getQValue(state, actions[0])
    bestAction = actions[0]
    for action in actions:
      q = self.getQValue(state, action)
      if q > value:
        bestAction = action
        value = q

    return bestAction


  def getPolicy(self, state):
    return self.computeActionFromQValues(state)

  def getValue(self, state):
    return self.computeValueFromQValues(state)



  def getWeights(self):
    return self.weights

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    features = self.getFeatures(state, action)

    sum = 0
    for feature, value in features.iteritems():
      sum += self.weights[feature] * value
    return sum

  def getFeatures(self, state, action):
    # extract the grid of food and wall locations and get the ghost locations
    food = self.getFood(state)
    walls = state.getWalls()
    myState = state.getAgentState(self.index)

    enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
    invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    ghosts = []
    if len(invaders) > 0:
      ghosts = [a.getPosition() for a in invaders]


    features = util.Counter()

    features["bias"] = 1.0

    # compute the location of pacman after he takes the action

    x, y = myState.getPosition()
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)

    # count the number of ghosts 1-step away
    features["#-of-ghosts-1-step-away"] = sum((next_x, next_y) in Actions.getLegalNeighbors(g, walls) for g in ghosts)

    # if there is no danger of ghosts then add the food feature
    if not features["#-of-ghosts-1-step-away"] and food[next_x][next_y]:
      features["eats-food"] = 1.0

    dist = closestFood((next_x, next_y), food, walls)
    if dist is not None:
      # make the distance a number less than one otherwise the update
      # will diverge wildly
      features["closest-food"] = float(dist) / (walls.width * walls.height)
    features.divideAll(10.0)
    return features


def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None


class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """



  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != self.nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def nearestPoint(self, pos):
    """
    Finds the nearest grid point to a position (discretizes).
    """
    (current_row, current_col) = pos

    grid_row = int(current_row + 0.5)
    grid_col = int(current_col + 0.5)
    return (grid_row, grid_col)

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
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class DefenceAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


