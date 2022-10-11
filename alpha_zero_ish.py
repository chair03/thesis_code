import numpy as np
from collections import defaultdict

class Branch:
  def __init__(self, prior):
    self.prior = prior
    self.visit_count = 0
    self.value = 0.0

class Node:
  def __init__(self,state,turns,value,action_probs,valid_actions,parent,last_move):
    self.state = state,turns
    self.parent = parent
    self.value = value 
    self.branches = defaultdict(int)
    self.total_visit_count = 1
    valid_probs = np.take(action_probs,valid_actions)
    valid_prob_sum = np.sum(valid_probs)

    for index,action in enumerate(valid_actions):
      prob = action_probs[index]/valid_prob_sum
      self.branches[action] = Branch(prob)
    self.children = {}
    self.last_move = None

  def actions(self):
    return self.branches.keys()
  
  def add_child(self,action,child):
    self.children[action] = child

  def value(self):
    if self.visit_count == 0:
      return 0
    else:
      return self.value/self.visit_count

    def prior(self,action):
    return self.branches[action].prior

  def child_exists(self,action):
    return action in self.children 

  def expected_value(self,action):
    branch = self.branches[action]
    if branch.visit_count == 0:
      return 0.0
    return branch.value/branch.visit_count

  def visit_count(self,action):
    if action in self.branches:
      return self.branches[action].visit_count
    return 0

  def record_visit(self,action,value):
    self.total_visit_count += 1
    self.branches[action].visit_count += 1
    self.branches[action].value += value

class AgentTree():

  def __init__(self,collector,cop_model,robber_model,c,num_simulations,is_cop):
    self.c = c
    self.cop_model = cop_model
    self.robber_model = robber_model
    self.num_simulations = num_simulations
    self.is_cop = is_cop
    self.collector = collector
  
  def save_model(self,path):
    self.model.save_weights(path)

  def load_model(self,path):
    self.model.load_weights(path)
  
  def select_branch(self,node):
    parent_visit = node.total_visit_count
    

    def ucb(action):
      Q = node.expected_value(action)
      P = node.prior(action)
      n = node.visit_count(action)
      return Q + self.c*P*np.sqrt(parent_visit)/(n+1)
    return max(node.actions(),key = ucb)
   
  def predict(self,game_state):
    to_play = game_state[0,-1]
    state = tf.convert_to_tensor(game_state)
    state = tf.expand_dims(state,axis=0)
    priors,value = None,None
    if to_play == 1:
      priors,value = self.cop_model.predict(state)
    elif to_play == -1:
      priors,value = self.robber_model.predict(state)
    else:
      raise ValueError("Last column pf game_state can only be 1 or -1, value is ", to_play)
    priors = priors[0]
    value = value[0][0]
    return priors,value

    
  def create_node(self,game_state,turns,action=None,parent=None):
    action_probs,value = self.predict(game_state)
    valid_next_moves  = game_logic.valid_actions(game_state)
    new_node = Node(game_state,turns,value,action_probs,valid_next_moves,parent,action)
    if parent is not None:
      parent.add_child(action,new_node)
    return new_node

  def select_action(self,game_state,turns):
    root = self.create_node(game_state,turns)

    for i in range(self.num_simulations):
      node = root 
      next_action = self.select_branch(node)
      while node.child_exists(next_action):
        node = node.get_child(next_action)
        next_action = self.select_branch(node)
      next_state,turns = game_logic.take_action(node.state[0],node.state[1],next_action)
      child_node = self.create_node(next_state,turns,parent = node)
      action = next_action
      value = -1*child_node.value
      while node is not None:
        node.record_visit(action,value)
        action = node.last_move
        node = node.parent
        value = -1*value
    
    to_play = game_state[0,-1] 
    visit_counts = None
    if to_play == 1:
      visit_counts = np.array([root.visit_count(action) for action in range(game_logic.num_cop_actions)])
      self.collector.record_decision(game_state,visit_counts)
    elif to_play == -1:
        visit_counts = np.array([root.visit_count(action) for action in game_logic.robber_actions])
        self.collector.record_decision(game_state,visit_counts)

    return max(root.actions(),key=root.visit_count)

class EpisodeExperienceCollector:
  def __init__(self):
    self.episode_states = []
    self.episode_visit_counts = []

  def record_decision(self,state,visit_counts):
    self.episode_states.append(state)
    self.episode_visit_counts.append(visit_counts)

  def complete_episode(self,reward):
    num_states = len(self.episode_states)
    self.episode_rewards = [reward for _ in range(num_states)]

class ExperienceCollector:
  def __init__(self):
    self.states = []
    self.visit_counts = []
    self.rewards = []

  def combine(self,experiences):
    for experience in experiences:
      self.rewards += experience.episode_rewards
      self.states += experience.episode_states
      self.visit_counts += experience.episode_visit_counts