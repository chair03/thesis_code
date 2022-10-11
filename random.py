import numpy as np
class Robber:
  def __init__(self,graph):
    self.graph = graph
  def first_move(self,cops_postions):
    self.position = np.random.choice(list(self.graph.nodes()))
  def move(self,cops_postions):
    nbrs = list(self.graph.neighbors(self.position))
    self.postion = np.random.choice(nbrs)

class Cop:
  def __init__(self,graph):
    self.graph = graph
    self.position = None
  def first_move(self):
    self.position = np.random.choice(list(self.graph.nodes()))
  def move(self,robber_position):
    nbrs = list(self.graph.neighbors(self.position))
    self.postion = np.random.choice(nbrs)