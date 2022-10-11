import abc
import numpy as np
import os

from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tf.keras.layers import Dense
from tf.keras import Model
from tf.keras.optimizers import Adam
from tf import convert_to_tensor

from scipy.signal import lfilter

class Agent(abc.ABC):
  @abc.abstractmethod
  def take_action(self):
    pass
  def pass_info(self):
    pass

class AgentRandom(Agent):
  def __init__(self,num_actions):
    self.num_actions = num_actions
  def take_action(self,state,valid_actions):
    return np.random.choice(valid_actions)

class AgentRL(Agent):
    def __init__(self,graph,alpha,model,path,k):
        self.graph = graph
        self.k = k 
        self.learning_rate = 0.005
        self.number_of_nodes = graph.number_of_nodes()
        self.num_states = graph.number_of_nodes()**k*(graph.number_of_nodes()+1)
        self.num_actions = graph.number_of_nodes()
        self.actions = np.arange(graph.number_of_nodes())
    
        
        self.states = [np.arange(-1,graph.number_of_nodes()) for _ in range(k+1)]
        self.states = self.cartesian_product(*self.states)

    
    
        self.checkpoint_path =  'Robber/Checkpoint/'
        #self.tensorboard = ModifiedTensorBoard(log_dir="logs/{}-{}".format(MODEL_NAME, int(time.time())))
        '''
        self.checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
                filepath=agent_path,
                save_weights_only=True,
                verbose=1
            )
        '''
        self.alpha = alpha
        self.beta =  0.1
        self.gamma = 0.99
        self.name='robber'
        self.create_model(model)

        self.enc = LabelBinarizer().fit(np.arange(0,graph.number_of_nodes()))
        self.initialize()
        self.rewards = []
        
    def cartesian_product(self,*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[...,i] = a
        return arr.reshape(-1, la)
    
    def memoise(self,state, action,reward):
        self.states_history.append(state)
        self.action_history.append(action)
        self.rewards_history.append(reward)
        
  
    def create_model(self,base_model):
        
        x = Dense(512,activation='linear')(base_model.layers[-1].output)
        actor = Dense(self.num_actions,activation='softmax')(x)
        critic = Dense(1,activation = 'linear')(x)
        
        self.model = Model(inputs=base_model.input,outputs=[actor,critic])
        self.model.compile(optimizer=Adam(learning_rate=self.alpha))
        
        #self.policy = tf.keras.Model(inputs=[base_model.input],outputs=outputs)
        
        if(os.path.exists(self.checkpoint_path+'/checkpoint')):
            #self.load()
            pass
        
    
    def get_probs(self, state):
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state,0)
        return self.model(state)

    def legal_actions(self,position):
        if (position == -1):
            return self.actions
        else:
            possible_actions = list(self.graph.neighbors(position))
            possible_actions.append(position)
            
            return np.array(possible_actions)
        
    def get_action_index(self,valid_actions):
        return np.isin(self.actions,valid_actions)
        
    def normalize(self,naieve_action_probs,position):
        valid_actions = self.legal_actions(position)
        
        valid_indices = self.get_action_index(valid_actions)
        naieve_action_probs = tf.squeeze(naieve_action_probs).numpy()
        action_probs = naieve_action_probs[valid_indices]
        
        action_probs/=np.sum(action_probs)
        return action_probs,valid_actions

    def take_action(self):
        naieve_action_probs,_ = self.get_probs(self.current_state)
        action_probs,valid_actions= self.normalize(naieve_action_probs,self.position)
        action = np.random.choice(valid_actions,p = action_probs)
        self.current_action = action
        self.prev_position = self.position
        self.position = action
        #print("Robber taking action ",self.current_action)
        return action

    def get_state(self,robber_position,cop_positions):
        if(self.k>1):
            return self.enc.transform((robber_position,*cop_positions)).T
        else:
            return self.enc.transform([robber_position,cop_positions]).T
    
    def get_current_state(self):
        return self.current_state
    
    def get_prev_state(self):
        return self.prev_state
    
    def get_action(self):
        return self.current_action
    
    def get_position(self):
        return self.position
    
    def discounted_rewards(self):
        r = self.rewards_history[::-1]
        a = [1, -self.gamma]
        b = [1]
        y = lfilter(b, a, x=r)
        return y[::-1]
    
    
    def learn(self, position,state,action,reward,next_state,done):
        state = convert_to_tensor([state],dtype = tf.float32)
        next_state = convert_to_tensor([next_state],dtype = tf.float32)
        reward = convert_to_tensor(reward,dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
                action_probs, critic_value = self.model(state)
                _,critic_value_next = self.model(next_state)

                critic_value = tf.squeeze(critic_value)
                critic_value_next = tf.squeeze(critic_value_next)

                #probs = self.normalize(action_probs,position)

                #probs = convert_to_tensor(probs)
                action_probs = tfp.distributions.Categorical(probs = action_probs)
                log_prob = action_probs.log_prob(action)

                td = reward+self.gamma*critic_value_next*(1-int(done))-critic_value
                actor_loss = -log_prob*td

                critic_loss = td**2

                total_loss = actor_loss+critic_loss
        grads = tape.gradient(total_loss,self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def save(self): 
        print("Saving Model")
        self.model.save_weights(self.checkpoint_path)
        
    def load(self):
        print("Loading Model")
        self.model.load_weights(self.checkpoint_path)
    
    def initialize(self):
        self.position = -1
        self.current_state = None
        self.current_action = None
        self.rewards_history = []
        self.action_history = []
        self.states_history = []
    
    def get_prev_position(self):
        return self.prev_position
    
    def update_state(self,cop_positions):
        #pdb.set_trace()
        self.prev_state = self.current_state
        if(self.k>1):
            self.current_state = self.enc.transform((self.position,*cop_positions)).T
            
        else:
            self.current_state = self.enc.transform([self.position,cop_positions]).T
    
    def remeber_rewards(self,reward):
        self.rewards.append(reward)