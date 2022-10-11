import numpy as np
import networkx as nx

class Agent:
  def __init__(self,graph:nx.Graph,states:np.array,actions:np.array,decay:float)->None:
    self.graph = graph
    self.actions = actions
    self.states = states
    num_actions = actions.shape[0]
    num_states = states.shape[0]
    self.Q = np.zeros((num_states,num_actions))
    self.alpha = 0.2
    self.decay = decay
    self.epsilon = 0.9
    self.gamma = 1
    self.prev_action = None
    self.current_state = None
    self.current_action = None

  def update_epsilon(self)->None:
      self.epsilon *= self.decay
  
  def argmax(self, l:np.array)->float:
    ties = []
    max_val = max(l)
    for i in range(len(l)):
        if (l[i] == max_val):
            ties.append(i)
    return np.random.choice(ties)

      
  def take_action(self,state:int,valid_actions:np.array)->int:
      action = None
      if (np.random.rand()<self.epsilon):
          action =  np.random.choice(valid_actions)    
      else:
          qs = self.Q[state,:]
          valid_actions = np.sort(valid_actions)
          action = self.argmax(np.take(qs,valid_actions))
          action = valid_actions[action]
      self.prev_action = action
      self.prev_state = state
      
  def update_Q(self,reward:float,done:bool,state:int = None)->None:
      self.Q[self.prev_state,self.prev_action] = reward + done*self.gamma*self.Q[self.current_state,self.current_action]