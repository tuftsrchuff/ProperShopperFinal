import numpy as np
# import pandas as pd
# import torch
# import torch.nn.functional as F
import json  # Import json for dictionary serialization

class QLAgent:
    # here are some default parameters, you can use different ones
    def __init__(self, action_space, alpha=0.5, gamma=0.8, epsilon=0.1, mini_epsilon=0.01, decay=0.999):
        self.action_space = action_space 
        self.alpha = alpha               # learning rate
        self.gamma = gamma               # discount factor  
        self.epsilon = epsilon           # exploit vs. explore probability
        self.mini_epsilon = mini_epsilon # threshold for stopping the decay
        self.decay = decay               # value to decay the epsilon over time
        # self.qtable = pd.DataFrame(columns=[i for i in range(self.action_space)])  # generate the initial table
        self.qtable = np.zeros((50,50),dtype=float)
    
    def trans(self, state, granularity=0.5):
        # You should design a function to transform the huge state into a learnable state for the agent
        # It should be simple but also contains enough information for the agent to learn
        player_pos = state["observation"]["players"][0]["position"]
        player_pos_flooring_with_granularity = np.floor([pos/granularity for pos in player_pos])
        state_trans = np.array(player_pos_flooring_with_granularity)
        return state_trans
        
    def learning(self, action, rwd, state, next_state):
        # implement the Q-learning function
        pass
    def exploring(self,action,rwd,state,next_state):
        # explore and map the env using intrinsic reward
        pass
    # choose action based on value table
    def choose_action(self, state):
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        randnum = np.random.uniform(0,1)
        actions = list(range(len(self.action_space)))
        if randnum <= self.epsilon:
            action = np.random.choice(actions)
            return action
        else:
            values = self.get_nearby_values(state)
            action = np.argmax(values)
            return action

    def get_nearby_values(self,state):
        values = []
        for i,j in zip([-1,1,0,0],[0,0,-1,1]):
            # if not exceed Q table, give value, else give 0 (states to be exlored)
            values.append(self.qtable[state[0]+i][state[1]+j] 
                            if not self.check_state_at_edge(state + np.array([i,j]))
                            else 0)
        return values

    def check_state_at_edge(self,state):
        if state[0] in range(0,self.qtable.shape[0]) and state[1] in range(0,self.qtable.shape[1]):
            return False
        else:
            return True
    def expand_qtable(self,state):
        qtable_shape = self.qtable.shape
        if state[0] >= qtable_shape[0] or state[1] >= qtable_shape[1]:
            new_shape = [max(qtable_shape[0],state[0]+10),max(qtable_shape[1],state[1]+10)]
            pad_width = [(0,new_shape[0]-qtable_shape[0]),(0,new_shape[1]-qtable_shape[1])]
            self.qtable = np.pad(self.qtable, pad_width=pad_width, mode='constant',constant_values=0)

             
        # implement the action selection for the fully trained agent
       
