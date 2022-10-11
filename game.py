import numpy as np
from networkx import Graph
class Game:
    def __init__(self,graph:Graph,max_turns:int,k:int,alpha:int,cops,robber,episodes:int)->None:
        self.graph = graph
        num_nodes = graph.number_of_nodes()
        self.k = k
        self.cop = cops
        self.robber = robber
        if k == 1: 
          self.cop_position = -1 
        else: 
          self.cop_position = -1*np.ones(k)

        self.robber_position = -1

        self.cop_actions = np.squeeze(np.array([np.arange(num_nodes) for _ in range(k)])) 
        if k > 1: self.cop_actions =  self.cartesian_product(*self.cop_actions)
        self.robber_actions = np.arange(num_nodes)
        self.max_turns = 50*num_nodes
        self.episodes = episodes


    def caught_robber(self)->bool:
        caught = self.cop_position == self.cop_position
        if (caught.any()):
            #print("Caught robber")
            self.cop_win = True
            self.stop = True
            return True
        return False

    
    def cartesian_product(self,*arrays)->np.array:
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a
        return arr.reshape(-1, la)

    def reset(self)->None:
        self.turn = 0
        self.cop_win = False
        self.stop = False
    
    def cop_action_index(self,action)->np.array:
        return np.where(np.all(action==self.cop_actions,axis=1))[0][0]

    def robber_action(self,action:int)->None:
      self.position = action
    
    def get_game_state(self):
      if (self.k == 1):
        return self.enc.transform((self.robber_position,self.cop_position)).T
      else:
        return self.enc.transform((self.robber_position,*self.cop_position)).T
    
    def cop_valid_actions(self):
        if (np.any(self.cop_position == -1)):
            return self.cop_actions
        else:
            if self.k == 1: 
                nbrs = list(self.graph.neighbors(self.cop_position))
                nbrs.append(self.cop_position)
                return np.array(nbrs)
            else:
                possible_actions = []
                for position in self.cop_position:
                    nbrs = list(self.graph.neighbors(self.cop_position))
                    nbrs.append(position)
                    possible_actions.append(nbrs)
                return np.array(np.meshgrid(*possible_actions)).T.reshape(-1,self.k)
      
    
    def get_action_indices(self,valid_actions):
        return np.isin(self.cop_actions,valid_actions)
    def robber_valid_actions(self):
      nbrs = list(self.graph.neighbors(self.robber_position))
      nbrs.append(self.robber_position)
      return np.array(nbrs)

    def cop_action(self,action_index)->None:
        action = self.cop_actions[action_index]
        self.cop_position = action 
    def tell_players(self)->None:
      self.cop.pass_info(self.stop)

      self.robber.pass_info(self.stop)
    
    def play(self)->None:
         self.reset()
         while (not self.stop and self.turn < self.max_turns):  
           self.turn += 1
           if (self.turn > 1):
               self.cop.pass_info(self.stop) 
           valid_action_indices = self.get_action_indices(self.cop_valid_actions())
           action_index = self.cop.take_action(self.get_game_state,valid_action_indices)
           self.cop_action(action_index)
           self.caught_robber()
           if (self.done):
             self.tell_players(self.stop)
           else:     
             if (self.turn > 1):
               self.robber.pass_info(self.stop) 
             action = self.robber.take_action(self.get_game_state),self.robber_valid_actions()
             self.caught_robber()
             if self.done:
               self.tell_players(self.stop)
             else:
               self.cop.pass_info(self.stop)    
    
    def run_epsiodes(self):
        for episode in range(0,self.episodes):
            self.current_episode = episode
            if(episode%100 == 0):
                print("episode ",episode)
            self.play    