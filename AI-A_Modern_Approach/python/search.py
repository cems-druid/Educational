import sys
from collections import deque

from utils import *

class Problem:
    """
    Abstract class for a formal problem. It should be used as subclass and implement the method actions.
    """

    def __init__(self, initial, goal=None):
        """
        General constructor specifies initial state and goal state
        """
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """
        Return the actions that can be executed in the given state. 
        The result is probably a list.
        If there are many elements, use yield rather than iteration. 
        """

        raise NotImplementedError

    def result(self, state, action):
        """
        Return the state that results from executing the given action in the given state.
        """

        raise NotImplementedError

    def goal_test(self, state):
        """
        Returns true if the state is a goal state
        """
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """
        state1 + action = state2 ==> cost (c) 
        """
        return c+1

    def value(self, state):
        """
        For optimization problems. Hill Climbing and related algorithms try to maximize this value.
        """

        raise NotImplementedError


class Node:
    """
    A node in the search-tree
    """

    def __init__(self, state, parent=None, action=None, path_cost=0):

        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """
        Lists the nodes that can be reached from the node.
        """
        return [self.child_node(problem, action) for action in problem.actions(self.state)]

    def child_node(self, problem, action):

        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """
        Return the sequence of actions to go from the root to this node
        """

        return [node.action for node in self.path()[1:]]

    def path(self):
        """
        Return a list of nodes forming the path from the root to this node.
        """
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))


    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        """
        Hash function is used inside of a node to see states that stored.
        """
        return hash(self.state)

    
class SimpleProblemSolvingAgentProgram:
    """
    Abstract framework for a problem-solving agent.
    """

    def __init__(self, initial_state=None):
        https://github.com/aimacode/aima-python/blob/master/search.py