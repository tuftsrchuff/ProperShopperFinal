import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import json  # Import json for dictionary serialization 
import os

class CartGetObjAgent:
    '''
    Agent gets certain object from a close location (using cart only).
    '''
    # here are some default parameters, you can use different ones
    def __init__(self, name, target, alpha=0.5, gamma=0.9, epsilon=0.05, mini_epsilon=0.05, decay=0.9999, granularity=0.2):
        self.name = name
        self.action_commands = ['NORTH', 'SOUTH', 'EAST', 'WEST', 'TOGGLE_CART', 'INTERACT']
        self.action_space = 5
        self.target = target
        self.alpha = alpha               # learning rate
        self.gamma = gamma               # discount factor  
        self.epsilon = epsilon           # exploit vs. explore probability
        self.mini_epsilon = mini_epsilon # threshold for stopping the decay
        self.decay = decay               # value to decay the epsilon over time
        self.qtable = pd.DataFrame(columns=[i for i in range(self.action_space)])  # generate the initial table 
        self.norm_table = pd.DataFrame(columns=[i for i in range(self.action_space)])
        self.granularity = granularity
    
    def trans(self, state, granularity=0.5):
        # You should design a function to transform the huge state into a learnable state for the agent
        #It should be simple but also contains enough information for the agent to learn
        player_info = state['observation']['players'][0]
        position = player_info['position'].copy()
        position[0] = int((1 + position[0]) / granularity)
        position[1] = int((1 + position[1]) / granularity)
        cart_position = state['observation']['carts'][0]['position'] 

        cart_position[0] = int((1 + cart_position[0]) / granularity)
        cart_position[1] = int((1 + cart_position[1]) / granularity) 

        holding = player_info['holding_food'] 
        if holding is not None and holding == self.target:
            holding = 1
        return json.dumps({'position': position, 'cart_position': cart_position, 'holding': holding}, sort_keys=True)
    
    def check_add(self, state):
        if self.trans(state) not in self.qtable.index:
            self.qtable.loc[self.trans(state)] = pd.Series(np.zeros(self.action_space), index=[i for i in range(self.action_space)])
            self.norm_table.loc[self.trans(state)] = pd.Series(np.zeros(self.action_space), index=[i for i in range(self.action_space)])
        
    def learning(self, action, rwd, state, next_state):
        # implement the Q-learning function
        self.check_add(state)
        self.check_add(next_state)
        obs = self.trans(state)
        next_obs = self.trans(next_state)
        td_error = rwd + self.gamma * self.qtable.loc[next_obs, :].max() - self.qtable.loc[obs, action] 
        self.qtable.loc[obs, action] += self.alpha * td_error 

        if len(next_state['violations']) != 0:
            self.norm_table.loc[obs, action] = 1 

    def choose_action(self, state):
        self.check_add(state)
        ob = self.trans(state)
        max_v = self.qtable.loc[ob, :].max()
        candidates = [i for i in range(self.action_space) if self.norm_table.loc[ob, i] != 1]
        if len(candidates) == 0: 
            candidates = range(self.action_space)
        greedy_candidates = [i for i in candidates if self.qtable.loc[ob, i] == max_v] 
        if len(greedy_candidates) == 0:
            greedy_candidates = candidates
        if self.epsilon > self.mini_epsilon:
            self.epsilon *= self.decay
        action = np.random.choice(candidates) if np.random.rand() < self.epsilon else np.random.choice(greedy_candidates)
        return action, None

    def save_qtables(self):
        self.qtable.to_json(self.name + '.json') 
        self.norm_table.to_json(self.name + '_norm.json')

    def load_qtables(self):
        # pass 
        if os.path.exists(self.name + '.json'):
            self.qtable = pd.read_json(self.name + '.json')
        if os.path.exists(self.name + '_norm.json'):
            self.norm_table = pd.read_json(self.name + '_norm.json')
        # self.qtable = pd.read_json(self.name + '.json')
        # self.norm_table = pd.read_json(self.name + '_norm.json')
