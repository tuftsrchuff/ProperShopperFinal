import numpy as np
# import torch
# import torch.nn.functional as F
import json  # Import json for dictionary serialization

# Just a standard Q learning, only difference is we have a explored_table
# to now which state is explored in the episode


class QLAgent:
    # here are some default parameters, you can use different ones
    def __init__(self, action_space, alpha=0.5, gamma=0.8, epsilon=0.1, mini_epsilon=0.01, decay=0.999,granularity=0.6):
        self.action_space = action_space 
        self.alpha = alpha               # learning rate
        self.gamma = gamma               # discount factor  
        self.epsilon = epsilon           # exploit vs. explore probability
        self.mini_epsilon = mini_epsilon # threshold for stopping the decay
        self.decay = decay               # value to decay the epsilon over time
        # self.qtable = pd.DataFrame(columns=[i for i in range(self.action_space)])  # generate the initial table
        # for now, we assume Q table is 50 by 50, we can always expand it later
        self.explored_table = np.zeros((100,100),dtype=int)
        self.q_table = np.zeros((50,50,action_space),dtype=float)
        self.granularity = granularity
    def trans(self, state):
        granularity = self.granularity
        # You should design a function to transform the huge state into a learnable state for the agent
        # It should be simple but also contains enough information for the agent to learn
        # we approxiamte position with granularity
        player_pos = state["observation"]["players"][0]["position"]
        player_pos_go_left_downward = np.floor([pos/granularity for pos in player_pos])
        # state_list = np.append(player_pos_go_left_downward)
        state_trans = np.array(player_pos_go_left_downward, dtype=int)
        return state_trans

    def explored(self,state):
        
        trans_state = self.trans(state)
        self.adapting_q_table(trans_state)
        if self.explored_table[trans_state[0],trans_state[1]] == 0:
            return 0
        else:
            return self.explored_table[trans_state[0],trans_state[1]]

    def set_explored(self,state):
        trans_state = self.trans(state)
        self.explored_table[trans_state[0],trans_state[1]] += 1


    def learning(self, action, rwd, state, next_state):
        # update Q table
        state_trans = self.trans(state)
        next_state_trans = self.trans(next_state)
        self.adapting_q_table(state_trans)
        self.adapting_q_table(next_state_trans)
        next_action = self.choose_action(next_state,use_epsilon=False)
        next_q_value = self.q_table[tuple(next_state_trans)][next_action]
        self.q_table[tuple(state_trans)][action] += self.alpha * (rwd + self.gamma*next_q_value - self.q_table[tuple(state_trans)][action])
        self.epsilon = self.epsilon * self.decay
        # ensure epsilon if greater than minimum value
        if self.epsilon < self.mini_epsilon:
            self.epsilon = self.mini_epsilon
        # implement the Q-learning function

    def choose_action(self, state, use_epsilon = True):
        # implement the action selection for the fully trained agent
        # epsilon greedy
        if use_epsilon == True:
            epsilon = np.random.uniform(0,1)
        else:
            epsilon = 1
        trans_state = self.trans(state)
        self.adapting_q_table(trans_state)
        if epsilon <= self.epsilon:
            # if epsilon, do random action
            action = np.random.choice(self.action_space)
        else:
            # if not epsilon, choose action with max Q 
            # print(trans_state)
            q_values = self.q_table[tuple(trans_state)]
            max_q = np.max(q_values)
            max_indices = np.where(q_values == max_q)[0].tolist()
            if len(max_indices) > 1:  
                return np.random.choice(max_indices)  
            else:  
                return max_indices[0] 
        # print(action) 
        return action
    def restore_explored_table(self):
        # self.explored_table = np.zeros((100,100),dtype=int)
        self.explored_table[:,:] = 0

    def adapting_q_table(self,state_trans):
        # check whether the state exceeds q table limit
        # if true, expand q table to adapt to state 
        q_table_shape = self.q_table.shape
        if state_trans[0] >= q_table_shape[0] or state_trans[1] >= q_table_shape[1]:
            new_shape = [max(q_table_shape[0],state_trans[0]+1),max(q_table_shape[1],state_trans[1]+1)]
            pad_width = [(0,new_shape[0]-q_table_shape[0]),(0,new_shape[1]-q_table_shape[1])] + [(0,0)]*(len(q_table_shape) - 2)
            print(pad_width)
            self.q_table = np.pad(self.q_table, pad_width=pad_width, mode='constant',constant_values=0)
            self.explored_table = np.pad(self.explored_table, pad_width=pad_width[:2], mode='constant',constant_values=0)


       
