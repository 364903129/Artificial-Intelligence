"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""
from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    # print("Start: %s" % (str(problem.startingState())))
    # print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    # print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    # create a stack
    statecontainer = Stack()
    # init
    visited = []
    statecontainer.push((problem.startingState(), []))
    (state, path) = statecontainer.pop()
    visited.append(problem.startingState())

    # while not goal
    while not problem.isGoal(state):
        # take the successor
        for currentstate in problem.successorStates(state):
            # if not visited or is goal
            if (not currentstate[0] in visited) or problem.isGoal(currentstate[0]):
                statecontainer.push((currentstate[0], path + [currentstate[1]]))
                # print("the current state is ", currentstate)
                visited.append(currentstate[0])  # add the visited state
        (state, path) = statecontainer.pop()
        # print("the path is ", path)

    return path


def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    # create a queue
    statecontainer = Queue()
    # init
    visited = []
    statecontainer.push((problem.startingState(), []))
    # print(problem.startingState())
    (state, path) = statecontainer.pop()
    visited.append(problem.startingState())

    # while not goal
    while not problem.isGoal(state):
        # take the successor
        for currentstate in problem.successorStates(state):
            # if not visited or is goal
            # print(currentstate[0])
            if (not currentstate[0] in visited) or problem.isGoal(currentstate[0]):
                statecontainer.push((currentstate[0], path + [currentstate[1]]))
                # dd the visited state
                visited.append(currentstate[0])
        if statecontainer.isEmpty():
            return path
        else:
            (state, path) = statecontainer.pop()
            # print(state)
    # print(path)
    return path


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    # create a priority queue
    statecontainer = PriorityQueue()
    # init
    visited = []
    # insert the starting point into the queue
    statecontainer.push((problem.startingState(), [], 0), 0)
    # while the queue is not empty
    while not statecontainer.isEmpty():
        # dequeue the maximum pirority element from the queue
        (state, path, cost) = statecontainer.pop()
        visited.append(state)
        # if the path is ending in the goal state, return the path and exit
        if problem.isGoal(state):
            # print("found the goal!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return path
        # else insert all the children of the dequeue element, with cost
        for currentstate in problem.successorStates(state):
            if currentstate[0] not in visited:
                statecontainer.push((currentstate[0], path + [currentstate[1]],
                                     currentstate[2] + cost),
                                    currentstate[2] + cost)
        # print(path)

    return path


def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    # create a priority queue
    statecontainer = PriorityQueue()
    # init
    visited = []
    # insert the starting point into the queue
    statecontainer.push((problem.startingState(), []),
                        0 + heuristic(problem.startingState(), problem))
    # while the queue is not empty
    while not statecontainer.isEmpty():
        # dequeue the maximum pirority element from the queue
        (state, path) = statecontainer.pop()
        # if the path is ending in the goal state, return the path and exit
        if problem.isGoal(state):
            # print("found the goal!!!!!!!!!!!!")
            return path
        if state not in visited:
            visited.append(state)
            # else insert all the children of the dequeue element, with cost
            for currentstate in problem.successorStates(state):
                if currentstate[0] not in visited:
                    # print(heuristic(currentstate[0], problem))
                    statecontainer.push((currentstate[0], path + [currentstate[1]]),
                                currentstate[2] + heuristic(currentstate[0],
                                problem) + problem.actionsCost(path))
            # print(problem.actionsCost(path))

    return path
