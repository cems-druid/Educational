from utils import distance_squared, turn_heading
from statistics import mean
from ipythonblock import BlockGrid
from IPython.display import HTML, display, clean_output

import random
import copy
import collections
import number

class Thing:

    def __repr__(self):
        """This represents any physical object that can appear in an Environment. You subclass
        Thing to get the things you want. Each thing can have a .__name__ slot """
        return '<{}>'.format(getattr(self, '__name__', self.__class__.__name__))
    
    def is_alive(self):
        """ Things that are 'alive' should return true """
        return hasattr(self, 'alive') and self.alive
        
    
    def show_state(self):
        """Display the agent's internal state. Subclasses should override."""
        print("I don't know how to show_state")
        
    def display(self, canvas, x, y, width, height):
        """Display an image of this Thing on the canvas"""
        pass
        

class Agent(Thing):
    """An agent is a subclass of Thing with one required instance attribute"""
    
    def __init__(self, program=None):
        self.alive = True
        self.bump = False
        self.holding = []
        self.performance = 0
        
        if program is None or not isinstance(program, collections.abc.Callable):
            print("Can't find a valid program for {}, falling back to default.".format(self.__class__.__name__))
            
            def program(percept):
                return eval(input('Percept={}; action?'.format(percept)))
                
        self.program = program


        def can_grab(self,thing):
            "Return True if this agent can grab anything"""
            return False
            
    def TraceAgent(agent):
    
        """Wrap the agent's program to print its input"""
        def new_program(percept):
            action = old_program(percept)
                
                
                
                
                
                
                
                
                
                
                
            
            